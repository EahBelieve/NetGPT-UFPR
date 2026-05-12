"""
Core pruner for NetGPT models.

Handles:
  - Identifying all prunable nn.Linear layers
  - Collecting activations (for Wanda) via forward hooks
  - Collecting gradients (for Pruner-Zero) via backward pass
  - Computing pruning masks from metric scores
  - Applying masks to zero out pruned weights
  - Reporting sparsity statistics

Key correction vs. v1:
  The original Wanda code (github.com/locuslab/wanda) uses PER-ROW pruning:
  each row of the weight matrix gets exactly sparsity% of its weights pruned.
  This ensures no output neuron is completely zeroed out.
  
  Our v1 used GLOBAL pruning (one threshold per layer), which can create
  dead neurons. This v2 matches Wanda's per-row approach.

Reference: wanda/lib/prune.py lines 150-165
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from pruning.metrics import METRICS


class NetGPTPruner:
    """
    Post-training pruner for the NetGPT Classifier model.
    
    Usage:
        pruner = NetGPTPruner(model, metric="wanda", sparsity=0.5)
        pruner.calibrate(calibration_loader, device, args)
        pruner.prune()
        stats = pruner.get_stats()
    """
    
    def __init__(self, model, metric="magnitude", sparsity=0.5,
                 prune_output_layers=False):
        """
        Args:
            model: Classifier instance (from run_understanding.py)
            metric: "magnitude" | "wanda" | "pruner_zero"
            sparsity: fraction of weights to zero out per row (0.0 to 1.0)
            prune_output_layers: if True, also prune output_layer_1/2
        """
        self.model = model
        self.metric_name = metric
        self.metric_fn = METRICS[metric]
        self.sparsity = sparsity
        self.prune_output_layers = prune_output_layers
        
        # Collect all prunable linear layers
        self.linear_layers = OrderedDict()
        self._collect_linear_layers()
        
        # Storage for calibration data
        self.activation_norms = {}  # layer_name -> tensor(in_features,)
        self.gradients = {}         # layer_name -> tensor(out_features, in_features)
        self.masks = {}             # layer_name -> binary tensor
        
        # Hooks
        self._hooks = []
    
    def _collect_linear_layers(self):
        """
        Find all nn.Linear layers to prune.
        
        Matches Wanda's find_layers() which recursively finds nn.Linear modules.
        We skip output classification layers (output_layer_1/2) unless requested,
        same as Wanda skips the lm_head.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if not self.prune_output_layers:
                    if "output_layer" in name:
                        continue
                self.linear_layers[name] = module
        
        print(f"[Pruner] Found {len(self.linear_layers)} prunable linear layers")
        for name, layer in self.linear_layers.items():
            print(f"  {name}: {layer.weight.shape}")
    
    def _register_activation_hooks(self):
        """
        Register forward hooks to collect activation norms for Wanda.
        
        Matches Wanda's WrappedGPT.add_batch() which accumulates:
            self.scaler_row += inp.float().pow(2).sum(dim=0)
        
        We do the same: accumulate X^2 per input dimension, then sqrt at the end.
        The Wanda code (lib/layerwrapper.py) stores scaler_row = sum(X_j^2),
        then in prune.py: W_metric = |W| * sqrt(scaler_row)
        """
        self._hooks = []
        self._activation_accum = {}
        self._activation_count = {}
        
        for name, module in self.linear_layers.items():
            self._activation_accum[name] = None
            self._activation_count[name] = 0
            
            def hook_fn(mod, input, output, layer_name=name):
                x = input[0].detach()
                if x.dim() == 3:
                    # (batch, seq_len, hidden) -> (batch*seq_len, hidden)
                    x = x.reshape(-1, x.shape[-1])
                # Accumulate X^2 per input dim (matches Wanda's scaler_row)
                x_sq = x.float().pow(2).sum(dim=0)  # (in_features,)
                if self._activation_accum[layer_name] is None:
                    self._activation_accum[layer_name] = x_sq
                else:
                    self._activation_accum[layer_name] += x_sq
                self._activation_count[layer_name] += x.shape[0]
            
            hook = module.register_forward_hook(hook_fn)
            self._hooks.append(hook)
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
    
    def _finalize_activation_norms(self):
        """
        Convert accumulated squared activations to norms.
        
        Wanda uses: sqrt(scaler_row) where scaler_row = sum(X^2)
        This gives the L2 norm of each input dimension across all calibration tokens.
        """
        for name in self.linear_layers:
            if self._activation_accum[name] is not None:
                # sqrt(sum(X_j^2)) = L2 norm, matching Wanda's sqrt(scaler_row)
                self.activation_norms[name] = torch.sqrt(
                    self._activation_accum[name]
                )
    
    def _collect_gradients(self, calibration_loader, device, args):
        """
        Collect gradients via backward pass on calibration data.
        
        Pruner-Zero Section 3.1: "we collect and preprocess gradient information
        using 128 calibration samples and archive it locally."
        """
        self.model.train()
        
        grad_accum = {name: torch.zeros_like(layer.weight)
                      for name, layer in self.linear_layers.items()}
        n_batches = 0
        
        for src_batch, tgt_batch, seg_batch, _ in calibration_loader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            seg_batch = seg_batch.to(device)
            
            self.model.zero_grad()
            loss, _ = self.model(src_batch, tgt_batch, seg_batch)
            if loss.dim() > 0:
                loss = loss.mean()
            loss.backward()
            
            for name, module in self.linear_layers.items():
                if module.weight.grad is not None:
                    grad_accum[name] += module.weight.grad.detach().clone()
            
            n_batches += 1
        
        for name in self.linear_layers:
            self.gradients[name] = grad_accum[name] / max(n_batches, 1)
        
        self.model.eval()
        print(f"[Pruner] Gradients collected over {n_batches} batches")
    
    def calibrate(self, calibration_loader, device, args):
        """
        Run calibration to collect statistics needed by the chosen metric.
        """
        print(f"[Pruner] Calibrating with metric='{self.metric_name}', "
              f"sparsity={self.sparsity}")
        
        if self.metric_name in ("wanda",):
            self._register_activation_hooks()
            self.model.eval()
            with torch.no_grad():
                for src_batch, tgt_batch, seg_batch, _ in calibration_loader:
                    src_batch = src_batch.to(device)
                    tgt_batch = tgt_batch.to(device)
                    seg_batch = seg_batch.to(device)
                    self.model(src_batch, tgt_batch, seg_batch)
            self._remove_hooks()
            self._finalize_activation_norms()
            print(f"[Pruner] Activation norms collected for "
                  f"{len(self.activation_norms)} layers")
        
        if self.metric_name in ("pruner_zero",):
            self._collect_gradients(calibration_loader, device, args)
        
        print("[Pruner] Calibration complete")
    
    def prune(self):
        """
        Compute and apply pruning masks based on the chosen metric.
        
        CRITICAL FIX (v2): Uses PER-ROW pruning to match Wanda's implementation.
        
        Wanda's code (lib/prune.py lines 156-164):
            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
            W_mask.scatter_(1, indices, True)
        
        This sorts EACH ROW independently and prunes the smallest sparsity%
        of weights per row. This guarantees every output neuron keeps exactly
        (1-sparsity)% of its input connections, preventing dead neurons.
        
        Our v1 used global per-layer thresholding which could empty entire rows.
        """
        print(f"[Pruner] Pruning {len(self.linear_layers)} layers "
              f"at {self.sparsity*100:.0f}% sparsity (per-row)...")
        
        total_params = 0
        total_pruned = 0
        
        for name, module in self.linear_layers.items():
            weight = module.weight.data
            
            # Build kwargs for the metric function
            kwargs = {}
            if name in self.activation_norms:
                kwargs["activation_norms"] = self.activation_norms[name]
            if name in self.gradients:
                kwargs["gradients"] = self.gradients[name]
            
            # Compute importance scores (same shape as weight)
            scores = self.metric_fn(weight, **kwargs)
            
            n_params = weight.numel()
            n_cols = weight.shape[1]  # in_features
            n_prune_per_row = int(n_cols * self.sparsity)
            
            if n_prune_per_row > 0:
                # ── PER-ROW PRUNING (matching Wanda) ──
                # Sort each row, get indices of the smallest scores
                sort_res = torch.sort(scores, dim=-1, stable=True)
                # Indices of the n_prune_per_row smallest scores per row
                prune_indices = sort_res[1][:, :n_prune_per_row]
                # Build mask: True where we KEEP the weight
                mask = torch.ones_like(weight)
                mask.scatter_(1, prune_indices, 0.0)
            else:
                mask = torch.ones_like(weight)
            
            # Apply mask in-place
            module.weight.data *= mask
            self.masks[name] = mask
            
            actual_pruned = (mask == 0).sum().item()
            total_params += n_params
            total_pruned += actual_pruned
        
        actual_sparsity = total_pruned / total_params if total_params > 0 else 0
        print(f"[Pruner] Done. Effective sparsity: {actual_sparsity*100:.2f}% "
              f"({total_pruned:,}/{total_params:,} weights zeroed)")
    
    def get_stats(self):
        """Return per-layer and global sparsity statistics."""
        stats = OrderedDict()
        total_params = 0
        total_zeros = 0
        
        for name, module in self.linear_layers.items():
            w = module.weight.data
            n = w.numel()
            z = (w == 0).sum().item()
            total_params += n
            total_zeros += z
            stats[name] = {
                "shape": tuple(w.shape),
                "total_params": n,
                "zero_params": z,
                "sparsity": z / n if n > 0 else 0
            }
        
        stats["__global__"] = {
            "total_params": total_params,
            "zero_params": total_zeros,
            "sparsity": total_zeros / total_params if total_params > 0 else 0,
            "metric": self.metric_name,
            "target_sparsity": self.sparsity
        }
        
        return stats
    
    def print_stats(self):
        """Pretty-print sparsity statistics."""
        stats = self.get_stats()
        print("\n" + "=" * 70)
        print(f"{'Layer':<45} {'Shape':>12} {'Sparsity':>10}")
        print("-" * 70)
        for name, s in stats.items():
            if name == "__global__":
                continue
            shape_str = f"{s['shape'][0]}x{s['shape'][1]}"
            print(f"  {name:<43} {shape_str:>12} {s['sparsity']*100:>8.1f}%")
        
        g = stats["__global__"]
        print("-" * 70)
        print(f"  {'GLOBAL':<43} {g['total_params']:>12,} {g['sparsity']*100:>8.2f}%")
        print(f"  Metric: {g['metric']}, Target: {g['target_sparsity']*100:.0f}%")
        print("=" * 70 + "\n")

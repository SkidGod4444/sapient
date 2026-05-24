//! Layout optimisation pass — detects NCHW vs NHWC preferences for Conv2d and
//! inserts Transpose nodes if the backend prefers a different layout.
//!
//! In Phase 1 this is a no-op (CPU uses NCHW); the pass is a hook for later
//! backends that prefer NHWC (e.g. TensorFlow-style Metal kernels).

use sapient_core::error::Result;
use crate::graph::Graph;
use crate::passes::Pass;

#[derive(Debug)]
pub struct LayoutOptimizationPass;

impl Pass for LayoutOptimizationPass {
    fn name(&self) -> &str { "layout-optimization" }

    fn run(&self, _graph: &mut Graph) -> Result<()> {
        // Phase 1 stub — CPU backend uses NCHW by convention, nothing to do.
        // Phase 5 (Metal): detect Metal Performance Shaders preference for NHWC
        // and insert Transpose(0,2,3,1) before Conv2d nodes.
        tracing::debug!("layout-optimization: pass (no-op in CPU phase)");
        Ok(())
    }
}

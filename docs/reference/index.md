# Reference

Additional reference documentation for JAX-FRC.

## Contents

- [Physics Concepts](physics.md) - Key physics background and equations
- [Model Comparison](comparison.md) - Feature comparison between models

## References

Theoretical framework based on FRC research and the following codes:

- **Lamy Ridge:** Resistive MHD code for FRC formation
- **NIMROD:** Extended MHD code for global stability
- **HYM:** Hybrid kinetic code for stability and beam physics

## Performance Considerations

- **JAX JIT Compilation:** All functions are JIT-compiled for optimal performance
- **GPU Acceleration:** Automatically uses GPU if available
- **Vectorization:** Operations are vectorized using JAX's array operations
- **Memory:** Hybrid kinetic model requires significant memory for particle data

## Contributing

Contributions are welcome! Areas for improvement:

- 3D geometry support
- More sophisticated boundary conditions
- Advanced particle weighting schemes
- Visualization tools
- Benchmarking against experimental data

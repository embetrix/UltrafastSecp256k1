---
name: Feature Request
about: Suggest a new feature, backend, or optimization
title: '[FEATURE] '
labels: 'enhancement'
assignees: ''

---

**Category**
- [ ] New backend (CUDA, Metal, ROCm, OpenCL, WASM, etc.)
- [ ] New curve / algorithm
- [ ] Performance optimization
- [ ] Build system / CI
- [ ] Documentation
- [ ] Other

**Target platform**
[e.g. x86-64, ARM64/Apple Silicon, RISC-V, ESP32, GPU]

**Describe the feature**
A clear description of what you want to add or improve.

**Motivation**
Why is this useful? What problem does it solve? Include benchmark data if relevant.

**Proposed implementation**
If you have ideas on how to implement this, describe the approach. Consider:
- Hot-path impact (allocation-free? branchless?)
- Memory model (scratch buffer? arena?)
- Which files/modules would be affected?

**Alternatives considered**
Other approaches you've thought about and why they're less suitable.

**Additional context**
Links, papers, reference implementations, or benchmark comparisons.

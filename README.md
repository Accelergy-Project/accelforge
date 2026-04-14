<div align="center">

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:004CFF,100:00CFFF&height=100&section=header&text=AccelForge&fontSize=52&fontColor=ffffff&fontAlignY=55" alt="AccelForge" />

***A framework to model and design tensor algebra accelerators.***

<br>

[![PyPI](https://img.shields.io/pypi/v/accelforge?style=for-the-badge&logo=pypi&logoColor=white&labelColor=3775A9&color=0B4F6C)](https://pypi.org/project/accelforge/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=1E415E)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-A31F34?style=for-the-badge&labelColor=2D2D2D)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/Docs-online-0A7BBB?style=for-the-badge&logo=readthedocs&logoColor=white&labelColor=1A1A1A)](https://accelergy-project.github.io/accelforge/)

[![CI](https://img.shields.io/github/actions/workflow/status/Accelergy-Project/accelforge/tests_and_publish.yaml?branch=main&style=for-the-badge&logo=githubactions&logoColor=white&label=CI&labelColor=24292F&color=2EA043)](https://github.com/Accelergy-Project/accelforge/actions)
[![Code style: black](https://img.shields.io/badge/code_style-black-000000?style=for-the-badge&logo=python&logoColor=white&labelColor=2D2D2D)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-2EA043?style=for-the-badge&logo=github&logoColor=white&labelColor=24292F)](https://github.com/Accelergy-Project/accelforge/pulls)

</div>

---

AccelForge models and designs tensor algebra accelerators. It uses [HWComponents](https://github.com/accelergy-project/hwcomponents) as a backend for area, energy, latency, and leak power estimates.

Learn more at the [website](https://accelergy-project.github.io/accelforge/) or on [GitHub](https://github.com/Accelergy-Project/accelforge).

## ⚡ Features

- **Flexible Full-Stack Modeling** of a wide variety of devices, circuits, architectures, workloads, and mappings. We integrate with [HWComponents](https://github.com/accelergy-project/hwcomponents), with easily-to-modify models for component area, energy, latency, and leak power.
- **Fast and optimal mapping** of workloads onto architectures, yielding the best-possible performance and energy efficiency.
- **Fusion-aware mapping** that optimizes fusion for cascades of Einsums, enabling end-to-end optimization of entire workloads.
- **Heterogenous Architectures** that can include multiple types of compute units.
- **Strong input validation** via Pydantic, with clear error reports for invalid specifications.
- **Pythonic Interfaces** that enable easy automation and integration with other tools.

## 📦 Install

```bash
pip install accelforge
```

## 🧪 Examples

See [`examples/`](examples) for architectures and workloads, and [`notebooks/`](notebooks) for tutorials.

## 📚 Cite

If you use AccelForge in your work, please see [Citing AccelForge](https://accelergy-project.github.io/accelforge/guide/citation.html) for the relevant papers.

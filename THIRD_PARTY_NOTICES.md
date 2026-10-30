# Third-party notices

Voice Workbench's own source is MIT licensed. The 0.3.0 Portable distribution includes a separate third-party audio runtime. Its components retain their individual licenses. Character weights and user media are not included in the public package.

- yt-dlp: https://github.com/yt-dlp/yt-dlp — upstream source is under the Unlicense. Its included components may have their own notices. The Windows build collects upstream package metadata and license files when available.
- RVC-Project WebUI: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI — MIT. The portable engine includes inference/training source and the accompanying notices. Foundation HuBERT/ContentVec, RMVPE and pretrained RVC assets originate from the RVC model repository: https://huggingface.co/lj1995/VoiceConversionWebUI (MIT repository metadata). No user-trained character weights or retrieval indices are included.
- ContentVec: https://github.com/auspicious3000/contentvec — MIT; notice copied to licenses/ContentVec-MIT.txt.
- Demucs: https://github.com/facebookresearch/demucs — MIT. The portable separator uses the upstream htdemucs checkpoint 955717e8-8726e21a.th, with its published filename checksum verified during assembly. It is stored locally; runtime model downloads are not needed. Source URL and full digest are in portable.json.
- PyMSS / PyMSS Core: https://github.com/pymss-project/pymss and https://github.com/pymss-project/pymss-core — MIT metadata and accompanying licenses retained with the installed packages. The older BS-RoFormer adapter remains available in source/external-engine mode; the portable distribution does not include ViperX BS-RoFormer weights.
- FFmpeg / FFprobe: https://ffmpeg.org/legal.html — separate executable processes. The included builds retain their LGPL/GPL notices; their version, build flags and license output are recorded under licenses/. Corresponding upstream source revisions are identified in the FFmpeg provenance file. They are not relicensed under Voice Workbench's MIT license.
- PyTorch, CUDA/cuDNN, Transformers, NumPy, SciPy, FAISS, SoundFile and the remaining runtime dependencies retain their upstream licenses, EULAs, package metadata and notices under engine/runtime/Lib/site-packages. The runtime is not wholly MIT licensed. A package inventory accompanies the portable distribution.
- Microsoft Visual C++ runtime: application-local redistributable runtime DLLs are included to avoid an additional installer; they retain Microsoft's applicable redistribution terms. Windows and the installed graphics driver are system components, not redistributed by this project.
- PyInstaller: https://pyinstaller.org/en/stable/license.html — build tool; its bootloader has a distribution exception. It is not the inference engine.
- CPython and Tcl/Tk: the Windows executable bundles the interpreter/GUI runtime via PyInstaller. Their license files are collected with the distribution.

This project does not grant rights to third-party models, downloaded videos, character identities, music or other input/output material.

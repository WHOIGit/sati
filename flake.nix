{
  description = "SATI: slice-aware training and inference for object detection in large images";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        # sati requires Python 3.12+
        python = pkgs.python312;
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            python
            pkgs.ruff
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
            # opencv-python wheels link against libGL, glib, and X client libs on Linux
            pkgs.libGL
            pkgs.glib
            pkgs.libxcb
            pkgs.libx11
            pkgs.libxext
            # shapely >= 2.0 ships its own GEOS in the wheel; no system geos needed
          ];

          # The trailing /run/opengl-driver/lib exposes the NVIDIA driver's
          # libcuda.so so pip-installed CUDA torch (ultralytics) finds the GPU.
          LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
            pkgs.libGL
            pkgs.glib
            pkgs.libxcb
            pkgs.libx11
            pkgs.libxext
          ]}:/run/opengl-driver/lib";

          shellHook = ''
            if [ ! -d .venv ]; then
              echo "Creating virtual environment..."
              ${python}/bin/python -m venv .venv
            fi
            source .venv/bin/activate
            # Prefer the Nix-native Ruff binary over PyPI's generic Linux wheel.
            export PATH="${pkgs.ruff}/bin:$PATH"

            # Install core + all non-torch extras (dev, synth) if not already present.
            if ! python -c "import sati" 2>/dev/null; then
              echo "Installing sati[dev,synth]..."
              pip install -e '.[dev,synth]' --quiet
            fi

            # Install ultralytics once torch is available (torch is not installed here
            # because the right CUDA wheel depends on your driver version).
            if python -c "import torch" 2>/dev/null; then
              if ! python -c "import ultralytics" 2>/dev/null; then
                echo "torch found — installing sati[ultralytics]..."
                pip install -e '.[ultralytics]' --quiet
              fi
            else
              echo ""
              echo "GPU step: install torch for your CUDA version, then re-enter the shell:"
              echo "  https://pytorch.org/get-started/locally/"
              echo "  pip install torch torchvision  # example; use the selector above"
              echo "ultralytics (YOLO training/inference) will be installed automatically once torch is present."
            fi

            echo ""
            echo "sati dev shell (Python $(python --version 2>&1 | cut -d' ' -f2))"
          '';
        };
      }
    );
}

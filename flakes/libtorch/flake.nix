{
  description = "LibTorch (PyTorch C++ library)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        libtorch-src =
          if pkgs.stdenv.isDarwin then
            pkgs.fetchzip {
              url = "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.4.0.zip";
              sha256 = "sha256-t/Z+ux4IpQ5Bhlb+I94NAyQ/kmZiVjF8adPMfpZdjaA="; # Placeholder - run 'nix build' to get the correct hash
              stripRoot = false;
            }
          else if pkgs.stdenv.isLinux then
            pkgs.fetchzip {
              url = "https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu121.zip";
              sha256 = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="; # Placeholder - run 'nix build' on Linux to get the correct hash
              stripRoot = false;
            }
          else
            throw "Unsupported platform";

        libtorch = pkgs.stdenv.mkDerivation {
          pname = "libtorch";
          version = "2.4.0";

          src = libtorch-src;

          dontBuild = true;
          dontConfigure = true;

          installPhase = ''
            mkdir -p $out
            cp -r libtorch/* $out/
          '';

          # Add metadata for easier discovery
          meta = with pkgs.lib; {
            description = "PyTorch C++ library";
            homepage = "https://pytorch.org/";
            license = licenses.bsd3;
            platforms = [ "x86_64-linux" "aarch64-darwin" ];
          };
        };

      in {
        packages = {
          default = libtorch;
          libtorch = libtorch;
        };

        # Development shell with libtorch available
        devShells.default = pkgs.mkShell {
          buildInputs = [ libtorch ];

          shellHook = ''
            export LIBTORCH="${libtorch}"
            export LIBTORCH_INCLUDE="${libtorch}/include"
            export LIBTORCH_LIB="${libtorch}/lib"

            # For Linux
            ${if pkgs.stdenv.isLinux then ''
              export LD_LIBRARY_PATH="${libtorch}/lib:$LD_LIBRARY_PATH"
            '' else ""}

            # For macOS
            ${if pkgs.stdenv.isDarwin then ''
              export DYLD_LIBRARY_PATH="${libtorch}/lib:$DYLD_LIBRARY_PATH"
            '' else ""}

            echo "LibTorch environment loaded!"
            echo "LIBTORCH: $LIBTORCH"
            echo "Headers: $LIBTORCH_INCLUDE"
            echo "Libraries: $LIBTORCH_LIB"
          '';
        };
      }
    );
}

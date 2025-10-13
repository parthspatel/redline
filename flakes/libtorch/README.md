# LibTorch Nix Flake

A Nix flake for installing LibTorch (PyTorch C++ library) on macOS (ARM64) and Linux (x86_64 with CUDA 12.6).

## Platforms

- **macOS ARM64**: CPU-only version (2.8.0)
- **Linux x86_64**: CUDA 12.6 version (2.8.0)

## First-time Setup

The SHA256 hashes are empty by default. Run the build once to get the correct hashes:

```bash
nix build
```

Nix will fail and provide the correct hash. Update the `sha256` field in `flake.nix` with the provided hash, then run again.

Repeat for both platforms if needed.

## Usage

### Install as a package

```bash
# Build the package
nix build

# Run from the result
./result
```

### Use in development shell

```bash
# Enter development shell with LibTorch available
nix develop

# The following environment variables are set:
# - LIBTORCH: Path to LibTorch installation
# - LIBTORCH_INCLUDE: Path to headers
# - LIBTORCH_LIB: Path to libraries
# - LD_LIBRARY_PATH (Linux) or DYLD_LIBRARY_PATH (macOS): Library paths
```

### Use in another flake

Add to your `flake.nix`:

```nix
{
  inputs = {
    libtorch.url = "path:/path/to/this/flake";
    # or if hosted on GitHub:
    # libtorch.url = "github:yourusername/libtorch-flake";
  };

  outputs = { self, nixpkgs, libtorch }: {
    # Use libtorch.packages.${system}.default in your derivation
  };
}
```

### Example: Building Rust project with LibTorch

```nix
devShells.default = pkgs.mkShell {
  buildInputs = [ libtorch.packages.${system}.default ];

  shellHook = ''
    export LIBTORCH="${libtorch.packages.${system}.default}"
    export LIBTORCH_INCLUDE="${libtorch.packages.${system}.default}/include"
    export LIBTORCH_LIB="${libtorch.packages.${system}.default}/lib"
  '';
};
```

## Environment Variables

When in the dev shell, the following variables are available:

- `LIBTORCH`: Root path to LibTorch installation
- `LIBTORCH_INCLUDE`: Path to include directory
- `LIBTORCH_LIB`: Path to lib directory
- `LD_LIBRARY_PATH` (Linux): Updated to include LibTorch libraries
- `DYLD_LIBRARY_PATH` (macOS): Updated to include LibTorch libraries

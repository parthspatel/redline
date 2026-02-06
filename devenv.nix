{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:
let
  pkgs-unstable = import inputs.nixpkgs-unstable { system = pkgs.stdenv.system; };
in
{
  env = {
  };

  packages = [
    # Core tools
    pkgs.git
    pkgs.jq
    pkgs.yq-go
    pkgs.curl
    pkgs.wget
    pkgs.ripgrep
    pkgs.fd
    pkgs.tree
    pkgs.just # command runner (Makefile alternative)

    pkgs.nixfmt-tree

    # Python tooling
    pkgs.python314
    pkgs.uv # fast Python package manager
    pkgs.ty
    pkgs.ruff
  ];

  # ── Python 3.14  ─────
  languages.python = {
    enable = true;
    package = pkgs.python314;
    uv = {
      enable = true;
      # sync.enable = true;
      package = pkgs.uv;
    };
  };

  # ── Rust ──────────
  languages.rust = {
    enable = true;
    toolchainFile = ./rust-toolchain.toml;
  };

  # ── Nix formatting ──────────────────────────────────────────
  languages.nix.enable = true;

  # Custom scripts
  scripts = {
    # Main dev command runner
    gsd.exec = ''
      npx get-shit-done-cc@latest "$@"
    '';

  };

  # Process management for local development
  processes = {
    # Local Kafka via docker-compose (if not using K8s)
    # kafka.exec = "docker-compose -f deploy/docker-compose.dev.yml up kafka";
  };

  # Git hooks
  difftastic.enable = true;
  git-hooks.hooks = {
    # Nix
    # nixfmt.enable = true;

    # Rust (tooling)
    rustfmt.enable = true;

    # Shell
    shfmt.enable = true;
    shellcheck.enable = true;

    # GitHub Actions
    action-validator.enable = true;

    # Conventional commits
    commitizen.enable = true;

    # Python
    ruff = {
      enable = true;
      entry = "uv run ruff check --fix";
      types = [ "python" ];
    };
  };

  # Claude Code integration
  claude.code = {
    enable = true;
    commands = {

    };
    hooks = {
      protect-secrets = {
        enable = true;
        name = "Protect sensitive files";
        hookType = "PreToolUse";
        matcher = "^(Edit|MultiEdit|Write)$";
        command = ''
          json=$(cat)
          file_path=$(echo "$json" | jq -r '.file_path // empty')

          if [[ "$file_path" =~ \.(env|secret)$ ]]; then
            echo "Error: Cannot edit sensitive files"
            exit 1
          fi
        '';
      };

      # Run tests after changes (PostToolUse hook)
      # test-on-save = {
      #   enable = true;
      #   name = "Run tests after edit";
      #   hookType = "PostToolUse";
      #   matcher = "^(Edit|MultiEdit|Write)$";
      #   command = ''
      #     # Read the JSON input from stdin
      #     json=$(cat)
      #     file_path=$(echo "$json" | jq -r '.file_path // empty')

      #     if [[ "$file_path" =~ \.rs$ ]]; then
      #       cargo test
      #     elif [[ "$file_path" =~ \.(ts|tsx)$ ]]; then
      #       pnpm test
      #     elif [[ "$file_path" =~ \.go$ ]]; then
      #       go test ./...
      #     elif [[ "$file_path" =~ \.java$ ]]; then
      #       sbt test
      #     elif [[ "$file_path" =~ \.py$ ]]; then
      #       uv run pytest
      #     fi
      #   '';
      # };
    };
  };
}

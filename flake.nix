{
  description = "A basic flake using pyproject.toml project metadata";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    # pre-commit-hooks.url = "github:cachix/pre-commit-hooks.nix";
    # flake-utils.url = "github:numtide/flake-utils";
    # my-lib.url = "github:zmrocze/nix-lib";
    
    pyproject-nix.url = "github:pyproject-nix/pyproject.nix";
    pyproject-nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    inputs@{ nixpkgs, pyproject-nix, flake-parts, ... }:
    let 
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
    in
    flake-parts.lib.mkFlake { inherit inputs; } {
      # imports = [
      #   ./pre-commit.nix
      #   my-lib.flakeModules.pkgs
      # ];
      # pkgsConfig.overlays = [
      #   my-lib.overlays.default
      # ];
      inherit systems;
      perSystem = { system, config, pkgs, ... }:
        let
          # Loads pyproject.toml into a high-level project representation
          # Do you notice how this is not tied to any `system` attribute or package sets?
          # That is because `project` refers to a pure data representation.
          project = pyproject-nix.lib.project.loadPyproject {
            # Read & unmarshal pyproject.toml relative to this project root.
            # projectRoot is also used to set `src` for renderers such as buildPythonPackage.
            projectRoot = ./.;
          };

          # We are using the default nixpkgs Python3 interpreter & package set.
          #
          # This means that you are purposefully ignoring:
          # - Version bounds
          # - Dependency sources (meaning local path dependencies won't resolve to the local path)
          #
          # To use packages from local sources see "Overriding Python packages" in the nixpkgs manual:
          # https://nixos.org/manual/nixpkgs/stable/#reference
          #
          # Or use an overlay generator such as uv2nix:
          # https://github.com/pyproject-nix/uv2nix
          python = pkgs.python3;

        in
        {
          # Create a development shell containing dependencies from `pyproject.toml`
          devShells.default =
            let
              # Returns a function that can be passed to `python.withPackages`
              arg = project.renderers.withPackages { inherit python; };

              # Returns a wrapped environment (virtualenv like) with all our packages
              pythonEnv = python.withPackages arg;

            in
            # Create a devShell like normal.
            pkgs.mkShell { packages = [ pythonEnv ]; };

          # Build our package using `buildPythonPackage

          packages = {
            # pykernel = pkgs.runCommand "pykernel" 
            #   {
            #     buildInputs = [ config.devShells.default ];
            #   } ''
            #     mkdir -p $out
            #     python3 -m ipykernel install --name "pykernel" --prefix $out
            #   '';
            default =
              let
                # Returns an attribute set that can be passed to `buildPythonPackage`.
                attrs = project.renderers.buildPythonPackage { inherit python; };
              in
              # Pass attributes to buildPythonPackage.
              # Here is a good spot to add on any missing or custom attributes.
              python.pkgs.buildPythonPackage (attrs);
          };
        };
    };
}
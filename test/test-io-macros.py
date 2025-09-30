from pyfreefem import FreeFemRunner
from os import getcwd
from os.path import join

macros_dir = join(getcwd(), "src/ff++/macros")


def run_script_test(script, params, macro_file, macros_dir=macros_dir):
    runner = FreeFemRunner(
        script,
        run_dir="test",
        macro_files=[join(macros_dir, macro_file)],
        debug=10,
    )

    exports = runner.execute(params, verbosity=0)
    return exports


script = """
 include "io-macros.edp";
 macroInputMeshParameters(meshParameters)
"""

params = {
    "meshKeys": '["cut", "r0", "fracRarc", "symmAxis"]',
    "meshValues": "[2*pi/3, 200, 0.1, pi/2.]",
}

mesh_params = run_script_test(script, params, "io-macros.edp")

print(mesh_params)

# = {
#     "Lc": 40,
#     "eta": 25000,
#     "xi": 0.1,
#     "zeta": -20,
#     "zi": 0.1,
# }

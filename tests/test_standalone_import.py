import importlib
import unittest


class TestStandaloneImport(unittest.TestCase):
    def test_standalone_package_imports_without_robojudo(self):
        pkg = importlib.import_module("sbc_log_plotter")
        io = importlib.import_module("sbc_log_plotter.sbc_log_io")
        gui = importlib.import_module("sbc_log_plotter.gui")

        self.assertIsNotNone(pkg)
        self.assertEqual(io.sbc_expected_columns(13), 159)
        self.assertNotIn("RoboJuDo", gui.build_arg_parser().description)

    def test_standalone_gui_has_only_sbc_mode(self):
        gui = importlib.import_module("sbc_log_plotter.gui")
        args = gui.build_arg_parser().parse_args(["--yaml", "policy.yaml", "--sbc-log", "Log.txt"])

        self.assertEqual(args.yaml, "policy.yaml")
        self.assertEqual(args.sbc_log, "Log.txt")
        self.assertFalse(hasattr(args, "mode"))

    def test_kp_kd_are_dimensionless(self):
        catalog = importlib.import_module("sbc_log_plotter.catalog")

        self.assertEqual(catalog.signal_unit("ref_kp"), "")
        self.assertEqual(catalog.signal_unit("ref_kd"), "")


if __name__ == "__main__":
    unittest.main()

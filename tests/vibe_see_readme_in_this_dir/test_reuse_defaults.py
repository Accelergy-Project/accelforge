"""
Tests for the Spatial.reuse / Spatial.may_reuse default behavior.

The defaults mirror Tensors.keep / Tensors.may_keep:
- If neither is set, reuse defaults to "Nothing" and may_reuse defaults to "All".
- If reuse is set, may_reuse defaults to the same value as reuse (so the spatial
  loop's eligible-for-reuse set equals its required-reuse set).
- If may_reuse is set explicitly, the user value is preserved.
"""

import unittest

from accelforge.frontend.arch import Spatial as ArchSpatial


class TestSpatialReuseDefaults(unittest.TestCase):
    def _eval(self, **kwargs):
        s = ArchSpatial(name="X", fanout=4, **kwargs)
        evaluated, _ = s._eval_expressions()
        return evaluated

    def test_neither_defined_may_reuse_is_all(self):
        e = self._eval()
        self.assertEqual(e.may_reuse, "All")

    def test_neither_defined_reuse_is_nothing(self):
        e = self._eval()
        self.assertEqual(e.reuse, "Nothing")

    def test_reuse_defined_may_reuse_defaults_to_reuse(self):
        e = self._eval(reuse="input")
        self.assertEqual(e.reuse, "input")
        self.assertEqual(e.may_reuse, "input")

    def test_reuse_defined_with_complex_set(self):
        e = self._eval(reuse="input | weight")
        self.assertEqual(e.reuse, "input | weight")
        self.assertEqual(e.may_reuse, "input | weight")

    def test_may_reuse_explicit_when_reuse_undefined(self):
        e = self._eval(may_reuse="output")
        self.assertEqual(e.reuse, "Nothing")
        self.assertEqual(e.may_reuse, "output")

    def test_may_reuse_explicit_nothing_when_reuse_undefined(self):
        e = self._eval(may_reuse="Nothing")
        self.assertEqual(e.reuse, "Nothing")
        self.assertEqual(e.may_reuse, "Nothing")

    def test_both_explicit_preserves_both(self):
        e = self._eval(reuse="input", may_reuse="All")
        self.assertEqual(e.reuse, "input")
        self.assertEqual(e.may_reuse, "All")

    def test_may_reuse_disjoint_from_reuse_when_both_explicit(self):
        e = self._eval(reuse="input", may_reuse="output")
        self.assertEqual(e.reuse, "input")
        self.assertEqual(e.may_reuse, "output")


class TestSpatialReuseDefaultsFromYaml(unittest.TestCase):
    """Verify YAML parsing applies the same defaults via Spec.from_yaml.

    These check the post-Spatial-eval values (sentinels resolved) before the
    full spec evaluation, so values are still the raw parsed strings."""

    def _spatial_from_yaml(self, spatial_yaml: str):
        import os
        import tempfile
        from pathlib import Path
        from accelforge.frontend.spec import Spec

        repo_root = Path(__file__).parent.parent.parent
        yaml_text = f"""
arch:
  nodes:
  - !Container
    name: PE
    spatial:
    - {spatial_yaml}
workload:
  rank_sizes: {{M: 8}}
  einsums:
  - name: E
    tensor_accesses:
    - {{name: a, projection: [m], bits_per_value: 8}}
    - {{name: b, projection: [m], output: True, bits_per_value: 8}}
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, dir=str(repo_root)
        ) as f:
            f.write(yaml_text)
            f.flush()
            try:
                spec = Spec.from_yaml(f.name)
                spatial = spec.arch.find("PE").spatial[0]
                evaluated, _ = spatial._eval_expressions()
                return evaluated
            finally:
                os.unlink(f.name)

    def test_yaml_no_reuse_no_may_reuse(self):
        s = self._spatial_from_yaml("{name: dim, fanout: 4}")
        self.assertEqual(s.reuse, "Nothing")
        self.assertEqual(s.may_reuse, "All")

    def test_yaml_reuse_only(self):
        s = self._spatial_from_yaml("{name: dim, fanout: 4, reuse: a}")
        self.assertEqual(s.reuse, "a")
        self.assertEqual(s.may_reuse, "a")

    def test_yaml_may_reuse_only(self):
        s = self._spatial_from_yaml("{name: dim, fanout: 4, may_reuse: b}")
        self.assertEqual(s.reuse, "Nothing")
        self.assertEqual(s.may_reuse, "b")


if __name__ == "__main__":
    unittest.main()

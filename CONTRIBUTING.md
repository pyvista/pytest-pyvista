## Contributing documentation for pytest-pyvista

### Release Procedure

#### Patch

Pull and checkout the latest release branch:

```bash
git checkout main
git pull
git checkout release/0.X  # where X is the most current release branch
git merge main
```

Now, update `__version__` within `pytest_pyvista/__init__.py` to the next patch. For example:

```py
__version__ = "0.3.2"
```

becomes

```py
__version__ = "0.3.3"
```

Commit these changes, tag, and push:

```bash
git commit -am "bump version to v0.3.3"
git push
git tag v0.3.3
git push --tags
```

### Minor

Pull and checkout the latest release branch:

On main, update `__version__` within `pytest_pyvista/__init__.py` to the next dev version. For example:

```bash
git checkout main
git pull
```

```py
__version__ = "0.4.dev0"
```

becomes

```py
__version__ = "0.5.dev0"
```

Commit and push (or open a PR):

```bash
git commit -am 'bumped main dev version'
git push
```

Once that's merged or committed, create a new release branch. This will be one
lower than the current dev version:

```bash
git checkout -b release/0.4
```

Finally, update the version in `pytest_pyvista/__init__.py`, commit, tag, and push:

```py
__version__ = "0.4.0"
```

Commit these changes, tag, and push:

```bash
git commit -am "bump version to v0.4.0"
git push
git tag v0.4.0
git push --tags
```

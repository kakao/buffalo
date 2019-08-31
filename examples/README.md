## Dockerfile.dev
This Dockerfile is provided for development environtmens of buffalo.

```
$> docker build -t buffalo.dev .
$> docker run -ti -e LC_ALL=C.UTF-8 buffalo.dev /bin/bash
$> cd /home/toros && source ./venv/bin/activate
```

External database are not included in this Dockerfile. For the unittest, checkout the ./tests/README.md.

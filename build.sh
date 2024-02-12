cp README.md public/README.md
cd public
pandoc -t revealjs -s -o index.html README.md -V revealjs-url=https://unpkg.com/reveal.js/ --include-in-header=slides.css -V theme=solarized --slide-level=3


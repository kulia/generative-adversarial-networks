make
sleep 0.1s
bibtex main.aux
sleep 1s
make force
sleep 1s
bibtex main.aux
sleep 1s
make force

cp main.pdf ../report_assignment_4.pdf
sleep 1s
make clean-all
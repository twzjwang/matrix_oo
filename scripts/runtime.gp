reset
set style fill solid
set grid
set term png enhanced font 'Verdana,10'
set output 'runtime.png'
set datafile separator ','
set xlabel "N"
set logscale xy

plot 'record.csv' using 1:2 with lines linewidth 2  title 'naive(ms)', \
'' using 1:3 with lines title 'sse(ms)', \
'' using 1:4 with lines title 'strassen(ms)'

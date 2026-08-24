# LEAD-0003

Q8_0 matrix-vector work is the largest traced GPU kernel family: 20.95% of
gfx1100 and 34.39% of gfx1201 kernel duration. The served streaming shapes are
narrow single-column projections with K/M pairs 1024/1024, 1024/4096, and
4096/1024.

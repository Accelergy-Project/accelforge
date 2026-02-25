def print_first_couple_rows(layout_func, nrows=4):
    W = 8
    first_couple_rows = [[None]*W for _ in range(nrows)]
    max_char_len = 0
    for m in range(M):
        for k in range(K):
            addr = layout_func(m, k)
            row = addr // W
            col = addr % W
            if row < nrows:
                element = f"A[{m},{k}]"
                max_char_len = max(max_char_len, len(element))
                first_couple_rows[row][col] = element

    # Align whitespace
    first_couple_rows = [
        [elem + " "*(max_char_len-len(elem)) for elem in row]
        for row in first_couple_rows
    ]

    for row in first_couple_rows:
        print(", ".join(row))


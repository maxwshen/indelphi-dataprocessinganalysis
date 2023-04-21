# Use to initialize, then input in seq2vec
def lib_to_fasta():
    names = []
    with open("../../data_libprocessing/names-libA.txt") as f:
        for line in f:
            names.append(line.strip("\n"))
    grna = []
    with open("../../data_libprocessing/grna-libA.txt") as f:
        for line in f:
            grna.append(line)
    with open("./grna-libA.fasta", "a") as f:
        for i in range(len(names)):
            f.write(f">{names[i]}\n")
            f.write(f"{grna[i]}\n")


if __name__ == "__main__":
    lib_to_fasta()

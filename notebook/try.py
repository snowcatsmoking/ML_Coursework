import nbformat

# 读取源notebook
with open('/Users/panmingh/Code/ML_Coursework/notebook/CNN.ipynb', 'r', encoding='utf-8') as f:
    cnn_nb = nbformat.read(f, as_version=4)

# 读取目标notebook
with open('/Users/panmingh/Code/ML_Coursework/notebook/CBU5201_miniproject_2526.ipynb', 'r', encoding='utf-8') as f:
    exp_nb = nbformat.read(f, as_version=4)

# 把cnn的所有cell（包括输出）追加到exp
exp_nb.cells.extend(cnn_nb.cells)

# 保存
with open('/Users/panmingh/Code/ML_Coursework/notebook/CBU5201_miniproject_2526.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(exp_nb, f)
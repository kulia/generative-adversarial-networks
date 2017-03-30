def write_variable_to_latex(var, var_name, filepath):
    filename = filepath + var_name + '.tex'
    target = open(filename, 'w')
    target.write(str(var))
    target.close()
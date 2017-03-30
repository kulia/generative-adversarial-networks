def write_variable_to_latex(var, var_name, filepath):
    filename = filepath + var_name + '.tex'
    target = open(filename, 'w')
    var = str(var)
    if len(var)>1:
        var = var.replace('  ', ', ')

    target.write(var)
    target.close()
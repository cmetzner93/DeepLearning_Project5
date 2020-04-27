def save_diagnostics_to_file(name_of_file, diagnostics):
    with open(name_of_file+'.txt', 'w') as f:
        f.writelines("%s\n" % batch for epoch in diagnostics for batch in epoch)
    f.close()

a = [10, 2]

b = [5, 4]

e = [3, 5]
r = [21, 4]

c = [a, b]
d = [e, r]
g = [c, d]

print(d)

save_diagnostics_to_file('test_file', g)


for epoch in g:
    print(epoch)
    for batch in epoch:
        print(batch)

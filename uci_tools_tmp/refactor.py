'''
I used ChatGPT to write this, so it's not necessarily organized in the most
efficient way, but it works. -PS
'''

import os
import sys
import nbformat
import argparse

def replace_in_notebook(notebook_path, old_string, new_string):
    with open(notebook_path, 'r') as f:
        nb = nbformat.read(f, as_version=4)
        for cell in nb.cells:
            if cell.cell_type == 'code':
                cell.source = cell.source.replace(old_string, new_string)
        with open(notebook_path, 'w') as f:
            nbformat.write(nb, f)

def replace_in_files_interactive(directory, old_string, new_string):
    script_file = os.path.abspath(__file__)
    for root, directories, files in os.walk(directory):
        directories[:] = [d for d in directories if not (d.startswith('.') 
                                                         or d == 'data'
                                                         or d.startswith('__'))
                         ]
        for filename in files:
            if (filename.startswith('.') 
                or filename.startswith('__')
                or (not filename.endswith('.py') 
                    and not filename.endswith('.jl')
                    and not filename.endswith('.toml')
                    and not filename.endswith('.md')
                    and not filename.endswith('.ipynb'))
               ):
                continue
            if filename == os.path.basename(__file__):
                continue
            file_path = os.path.join(root, filename)
            if os.path.isfile(file_path):
                print(f"Inspecting {file_path}")
                if filename.endswith('.ipynb'):
                    with open(file_path, 'r') as f:
                        nb = nbformat.read(f, as_version=4)
                        for cell in nb.cells:
                            if cell.cell_type == 'code':
                                if old_string in cell.source:
                                    index = cell.source.index(old_string)
                                    lines = cell.source.split('\n')
                                    line_number = 0
                                    for line in lines:
                                        if old_string in line:
                                            start = max(0, line_number - 2)
                                            end = min(len(lines), 
                                                      line_number + 3)
                                            print('\nOld string found!')
                                            print(f"File: {file_path}")
                                            print(f"Old String: {old_string}")
                                            print(f"New String: {new_string}")
                                            print("Context:")
                                            for j in range(start, end):
                                                print(lines[j].rstrip())
                                            print("------------")
                                            approval = input(
                                                "Do you approve this"
                                                " replacement? (y/n/q): ")
                                            if approval.lower() == 'y':
                                                cell.source = \
                                                        cell.source.replace(
                                                            old_string, 
                                                            new_string)
                                            elif approval.lower() == 'n':
                                                continue
                                            elif approval.lower() == 'q':
                                                sys.exit()
                                            else:
                                                print("Invalid input."
                                                      " Skipping this"
                                                      " instance.")
                                        line_number += 1
                    with open(file_path, 'w') as f:
                        nbformat.write(nb, f)
                else:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    for i, line in enumerate(lines):
                        if old_string in line:
                            start = max(0, i - 2)
                            end = min(len(lines), i + 3)
                            print('\nOld string found!')
                            print(f"File: {file_path}")
                            print(f"Old String: {old_string}")
                            print(f"New String: {new_string}")
                            print("Context:")
                            for j in range(start, end):
                                print('{0} {1}'.format(j + 1,
                                                       lines[j].rstrip()))
                            print("------------")
                            approval = input(
                                "Do you approve this replacement? (y/n/q): "
                            )
                            if approval.lower() == 'y':
                                lines[i] = lines[i].replace(old_string, 
                                                            new_string)
                                with open(file_path, 'w') as f:
                                    f.write(''.join(lines))
                            elif approval.lower() == 'n':
                                continue
                            elif approval.lower() == 'q':
                                sys.exit()
                            else:
                                print("Invalid input. Skipping this instance.")

def main():
    parser = argparse.ArgumentParser(description='Replace strings in files.')
    parser.add_argument('-o', '--old-string', type=str, required=True, 
                        help='Old string to replace.')
    parser.add_argument('-n', '--new-string', type=str, required=True, 
                        help='New string to replace with.')
    args = parser.parse_args()

    directory = '.'
    old_string = args.old_string
    new_string = args.new_string

    replace_in_files_interactive(directory, old_string, new_string)

if __name__ == '__main__':
    main()

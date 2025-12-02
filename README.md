# uci-tools
General-purpose code expected to be useful for multiple group members.

Install in your active environment with `pip install .`.

## `config.ini`
The user should put a `config_<environment>.ini` in their home directory for certain methods that save data in this package to function properly. The user must name it according to which conda environment they plan to use: `config_<environment>.ini`. If you use the `base` environment (which you shouldn't), you can name the config file `config.ini`. It should include a uci_tools_paths section as follows
```
[uci_tools_paths]
data_dir = /direc/containing/data/for/analysis/that/you/personally/generated
output_dir = /direc/where/code/should/save/output/such/as/figures/or/analyzed/data
snap_times = /path/to/snapshot/times/file
sat_image_dir = /direc/where/you/access/Courtneys/sat/images
host_iamge_dir = /direc/where/you/access/Courtneys/host/images
firebox_data_dir = /direc/containing/FIREBox/data/for/analysis
host_2d_shapes = /path/to/Courtneys/host/shape/results/file
host_2d_shapes = /path/to/Courtneys/sat/shape/results/file
```
If you don't create this and/or you don't have this section or any of its required entries before you import `uci_tools`, the package will create the missing requirements for you, and it will default to saving all outputs in `$HOME/output_<environment>/`.

## Using `refactor` script
Use `refactor` in the terminal by navigating to your project folder and then executing 

~~~
uci-tools-refactor -o old_text -n new_text
~~~
This will search all .py and .ipynb files in the project folder for `old_text`. `refactor` will give the user a preview of each instance where it finds `old_text` and ask for confirmation before replacing it with `new_text`.

# sims = {k: v[0] for k, v in sims.items()}
# sims = {}
# for v in sims2.values():
#     sims.update(v[0])
# for eisnum_name, sims2 in sims.items():
#     print(eisnum_name)
#     for s in sims2:
#         print(f'\t{s.compatibility_str()}')

# apt-get install npm
# git clone https://github.com/mermaid-js/mermaid.git
# cd mermaid
# npm install
# npm run build
# cd ..
# python3 -m http.server

# npm install -g @mermaid-js/mermaid-cli
# mmdc -i input.mmd -o output.svg

# pr.disable()
# s = io.StringIO()
# ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
# ps.print_stats(30)  # Print top 30 time-consuming functions
# print(s.getvalue())

# TODO: Check for ranks not in the mapping and put them at the bottom
# TODO: What if there are no loops? 
# TODO: Set _must_exist for all backing storage nodes
# TODO: Constraint attacher
# TODO: Can't have tile size constraints on backing memory
# TODO: Einsum orders
# TODO: Copy Einsums
# TODO: Test dataflow constraints and order of storage nodes
# TODO: Fix Einsum A loops in LoopForest mixed-shape matmuls figure
# TODO: Something we really really need to drive home for end users: Mapspace size increases exponentially with number of storage nodes
# I'm doing the tile shape exploration now and I'm trying to understand this note. I think I understand what you're saying.
# Can I ask one thing from the constraint code? If the constraint is an equality, then just set the tile_shape attribute of the node (or factor or whatever is needed) to the value.
# The tile shape exploration assumes a particular mapspace (in most cases, tile shapes are factors of the full rank shape), so an equality may never be satisfied. E.g., if the constraint sets the tile shape equal to a non-factor value because you want a particular imperfect factorization, but that's never in the mapspace, then you'll get nothing.
# It's also a bit more efficient to just set the value and the explorer doesn't have to figure out the equality by trial-and-error. For other more complicated constraints, trial-and-error is better.

# to_join_2 = copy.deepcopy(to_join)
# einsum = "QK"
# i = 0
# decompress_sims(to_join_2[einsum][i].mappings, decompress_data, [einsum])
# to_join_2[einsum][i].mappings.data
# # for s in to_join_2[einsum][i].mappings.data[f"{einsum}___MAPPING"][1].nodes:
# #     print(s)
# to_join_2[einsum][i].mappings.data

# # pip3 install plotly pydot ipywidgets anywidget
# import copy
# import re
# from fastfusion.frontend.mapping import Iteration, Mapping, Nested, Split, Storage
# from fastfusion.visualization.interactive import plotly_show
# from fastfusion.mapper.FFM.visualization import make_mapping
# from IPython.display import SVG

# # plotly_show(mappings.data, "RESOURCE_GlobalBuffer_LEVEL_0", "metric_energy", logscales=True)

# newmapping = make_mapping(mappings.data.iloc[0], spec.workload.einsum_names)
# display(SVG(newmapping.render()))

# # https://github.com/nodesource/distributions/issues/1157#issuecomment-849595760
# # cd /etc/apt/sources.list.d 
# # rm nodesource.list
# # apt --fix-broken install
# # apt update
# # apt remove nodejs
# # apt remove nodejs-doc
# # curl -fsSL https://deb.nodesource.com/setup_21.x | bash -
# # apt-get install --upgrade -y nodejs
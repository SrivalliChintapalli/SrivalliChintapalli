module tree
import Pkg
#Pkg.add("Collections")
Pkg.add("DecisionTree")
Pkg.add("DataStructures")



#using Collections
using DecisionTree
using DataStructures

struct Tree
    root::Any
end

function Tree(root)
    return Tree(root)
end

function prune(tree::Tree)
    bounds = DefaultDict(() -> [-Inf, Inf])
    tree.root = prune(tree.root, bounds)
end

function act(tree::Tree, observation)
    return act(tree.root, observation)
end

function to_string(tree::Tree, feature_names, action_names)
    return to_string(tree.root, 0, feature_names, action_names)
end

function to_graphviz(tree::Tree, feature_names, action_names;
                     integer_features=nothing, colors=nothing, fontname="helvetica")

    if integer_features === nothing
        integer_features = falses(length(feature_names))
    end

    if colors === nothing
        palette = [
            "#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc",
            "#ca9161", "#fbafe4", "#949494", "#ece133", "#56b4e9"
        ]
        colors = [palette[i % length(palette) + 1] for i in 1:length(action_names)]
    end

    header = "digraph Tree {\nnode [shape=box, style=\"filled, rounded\", color=\"gray\", fillcolor=\"white\", fontname=\"$fontname\"] ;\nedge [fontname=\"$fontname\"] ;\n"
    body = to_graphviz(tree.root, feature_names, action_names, integer_features, colors, 0)[1]
    footer = "}"
    return header * strip(body) * footer
end

function count_nodes(tree::Tree)
    return count_nodes(tree.root)
end

function count_depth(tree::Tree)
    return count_depth(tree.root)
end

Base.show(io::IO, tree::Tree) = print(io, to_string(tree, nothing, nothing))

struct TreeNode
    feature::Int
    threshold::Float64
    left_child::Any
    right_child::Any
end

function TreeNode(feature, threshold, left_child, right_child)
    return TreeNode(feature, threshold, left_child, right_child)
end

function act(node::TreeNode, observation)
    if observation[node.feature] <= node.threshold
        return act(node.left_child, observation)
    else
        return act(node.right_child, observation)
    end
end

function prune(node::TreeNode, bounds)
    old_bound = bounds[node.feature][2]
    bounds[node.feature][2] = node.threshold
    node.left_child = prune(node.left_child, bounds)
    bounds[node.feature][2] = old_bound

    old_bound = bounds[node.feature][1]
    bounds[node.feature][1] = node.threshold
    node.right_child = prune(node.right_child, bounds)
    bounds[node.feature][1] = old_bound

    if bounds[node.feature][1] > node.threshold
        return node.right_child
    end

    if bounds[node.feature][2] <= node.threshold
        return node.left_child
    end

    if isa(node.left_child, TreeLeaf) && isa(node.right_child, TreeLeaf) && node.left_child.action == node.right_child.action
        return node.left_child
    end

    return node
end

function to_string(node::TreeNode, depth, feature_names, action_names)
    left_string = to_string(node.left_child, depth + 1, feature_names, action_names)
    right_string = to_string(node.right_child, depth + 1, feature_names, action_names)
    padding = "    " ^ depth
    if feature_names === nothing
        return "$padding if X[$node.feature] <= $node.threshold:\n$left_string\n$padding else:\n$right_string"
    else
        return "$padding if $(feature_names[node.feature]) <= $node.threshold:\n$left_string\n$padding else:\n$right_string"
    end
end

function to_graphviz(node::TreeNode, feature_names, action_names, integer_features, colors, node_id)
    left_id = node_id + 1
    left_dot, new_node_id = to_graphviz(node.left_child, feature_names, action_names, integer_features, colors, left_id)
    right_id = new_node_id + 1
    right_dot, new_node_id = to_graphviz(node.right_child, feature_names, action_names, integer_features, colors, right_id)

    if node_id == 0
        edge_label_left = "yes"
        edge_label_right = "no"
    else
        edge_label_left = ""
        edge_label_right = ""
    end

    feature_name = feature_names[node.feature]

    split_condition = ifelse(integer_features[node.feature], Int(node.threshold), round(node.threshold, digits=3))
    predicate = "$node_id [label=\"if $feature_name <= $split_condition\"] ;\n"
    yes = left_id
    no = right_id
    edge_left = "$node_id -> $yes [label=\"$edge_label_left\", fontcolor=\"gray\"] ;\n"
    edge_right = "$node_id -> $no [label=\"$edge_label_right\", fontcolor=\"gray\"] ;\n"

    return "$predicate$left_dot$right_dot$edge_left$edge_right", new_node_id
end

function count_nodes(node::TreeNode)
    return 1 + count_nodes(node.left_child) + count_nodes(node.right_child)
end

function count_depth(node::TreeNode)
    return 1 + max(count_depth(node.left_child), count_depth(node.right_child))
end

struct TreeLeaf
    action::Int
end

function TreeLeaf(action)
    return TreeLeaf(action)
end

function act(leaf::TreeLeaf, _)
    return leaf.action
end

function to_string(leaf::TreeLeaf, depth, _, action_names)
    padding = "    " ^ depth
    if action_names === nothing
        return "$padding return '$leaf.action'"
    else
        return "$padding return '$(action_names[leaf.action])'"
    end
end

function to_graphviz(leaf::TreeLeaf, feature_names, action_names, integer_features, colors, node_id)
    label = action_names[leaf.action]
    color = colors[leaf.action]
    return "$node_id [label=\"$label\", fillcolor=\"$color\", color=\"$color\", fontcolor=white] ;\n", node_id
end

function prune(leaf::TreeLeaf, _)
    return leaf
end

function count_nodes(leaf::TreeLeaf)
    return 0
end

function count_depth(leaf::TreeLeaf)
    return 0
end

function sklearn_to_omdt_tree(tree::DecisionTreeClassifier)
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    value = tree.tree_.value

    function sklearn_to_omdt_tree_rec(node_i)
        if children_left[node_i] == children_right[node_i]
            return TreeLeaf(argmax(value[node_i][1]))
        end

        left_child = sklearn_to_omdt_tree_rec(children_left[node_i])
        right_child = sklearn_to_omdt_tree_rec(children_right[node_i])
        return TreeNode(feature[node_i], threshold[node_i], left_child, right_child)
    end

    return Tree(sklearn_to_omdt_tree_rec(1))
end
   
end





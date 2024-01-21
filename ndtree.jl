using Statistics

using Distances

include("MOKP.jl")

struct NDTreeNode
    points::Array{Pareto}
    approxIdealPoint::Array{Int}  # Approximate ideal point of this node
    approxNadirPoint::Array{Int}  # Approximate nadir point of this node
    children::Array{NDTreeNode}  # Children of this node
    parent::Union{NDTreeNode,Nothing}  # Parent of this node
end

const MAX_LEAF_SIZE = 20
# objectives dimension + 1
const REQUIRED_CHILDREN_COUNT = 4

# Algorithm 3: ND-Tree-based update
function update_archive!(archive::NDTreeNode, point::Pareto)
    if isempty(archive.children) && isempty(archive.points)
        # archive.points = [deepcopy(point)]
        # archive.approxIdealPoint = copy(point.objectives)
        # archive.approxNadirPoint = copy(point.objectives)
        push!(archive.points, deepcopy(point))
        for i in 1:length(archive.approxIdealPoint)
            archive.approxIdealPoint[i] = point.objectives[i]
            archive.approxNadirPoint[i] = point.objectives[i]
        end
        return true
    else
        if update_node!(archive, point)
            insert_node!(archive, point)
            return true
        end
    end
    return false
end

function update_node!(node::NDTreeNode, point::Pareto)
    # Check if the point is dominated by the approximate nadir point of the node (Property 1)

    if all(node.approxNadirPoint .>= point.objectives)
        # The point is rejected
        return false
        # Check if the point dominates the approximate ideal point of the node (Property 2)
    elseif all(point.objectives .>= node.approxIdealPoint)
        # Remove node and its whole sub-tree
        # node.points = Pareto[]
        # node.children = NDTreeNode[]
        empty!(node.points)
        empty!(node.children)
        return true
        # Check other conditions (Property 3)
    elseif any(node.approxIdealPoint .>= point.objectives) || any(point.objectives .>= node.approxNadirPoint)
        if isempty(node.children)
            # If the node is a leaf, check all points in the node
            to_remove = Pareto[]
            for z in node.points
                if all(z.objectives .>= point.objectives)
                    # Point z dominates the candidate point
                    return false
                elseif all(point.objectives .> z.objectives)
                    # add z to the set of points to be removed
                    push!(to_remove, z)
                end
            end
            # Remove all points dominated by the candidate point
            setdiff!(node.points, to_remove)
        else
            # If the node is an internal node, recursively check each child
            child_to_remove = NDTreeNode[]
            for child in node.children
                if !update_node!(child, point)
                    return false
                elseif isempty(child.points) && isempty(child.children)
                    # Add child to the set of children to be removed
                    push!(child_to_remove, child)
                end
            end
            setdiff!(node.children, child_to_remove)
            # Check if there's only one child remaining
            if length(node.children) == 1
                # Replace the current node with its single child
                new_points = node.children[1].points
                new_approxIdealPoint = node.children[1].approxIdealPoint
                new_approxNadirPoint = node.children[1].approxNadirPoint
                new_children = node.children[1].children
                new_node = NDTreeNode(new_points, new_approxIdealPoint, new_approxNadirPoint, new_children, node.parent)
                if !isnothing(node.parent)
                    push!(node.parent.children, new_node)
                    setdiff!(node.parent.children, [node])
                else
                    for i in 1:length(new_node.approxIdealPoint)
                        node.approxIdealPoint[i] = new_node.approxIdealPoint[i]
                        node.approxNadirPoint[i] = new_node.approxNadirPoint[i]
                    end

                    empty!(node.points)
                    empty!(node.children)
                    for child in new_node.children
                        push!(node.children, child)
                    end
                    for point in new_node.points
                        push!(node.points, point)
                    end
                    # println("replace root")
                end
                # println("replace")
            end
        end

        # Property 3 
    end
    return true
end

function insert_node!(node::NDTreeNode, point::Pareto)
    # If the node is a leaf node
    if isempty(node.children)
        push!(node.points, deepcopy(point))
        update_ideal_nadir!(node, point)


        if length(node.points) > MAX_LEAF_SIZE
            # Split the node if it exceeds the maximum size
            split_node!(node)
        end
    else
        # If the node is an internal node, find the closest child
        closest_child = find_closest_child(node, point)
        # Recursively insert the point into the closest child
        insert_node!(closest_child, point)
    end
end

function update_ideal_nadir!(node::NDTreeNode, point::Pareto)
    updated = false
    # Update the approximate ideal point if necessary
    for i in 1:length(node.approxIdealPoint)
        # println(point.objectives[i])
        # println(node.approxIdealPoint[i])
        if point.objectives[i] > node.approxIdealPoint[i]
            node.approxIdealPoint[i] = point.objectives[i]
            updated = true
        end
    end

    # Update the approximate nadir point if necessary
    for i in 1:length(node.approxNadirPoint)
        if point.objectives[i] < node.approxNadirPoint[i]
            node.approxNadirPoint[i] = point.objectives[i]
            updated = true
        end

    end

    # If there's an update and the node is not the root, update the parent node
    if updated && !isnothing(node.parent)
        update_ideal_nadir!(node.parent, point)
    end

end


function find_closest_child(node::NDTreeNode, point::Pareto)
    # Calculate the middle point between the ideal and nadir points
    middle_point = (node.approxIdealPoint + node.approxNadirPoint) / 2
    # Initialize variables to store the closest child and its distance
    closest_child = node.children[1]
    closest_distance = euclidean(point.objectives, middle_point)
    # Iterate over all children to find the closest one
    for child in node.children[2:end]
        # Calculate the distance to the middle point of the child
        distance = euclidean(point.objectives, (child.approxIdealPoint + child.approxNadirPoint) / 2)
        # Update the closest child and distance if a closer child is found
        if distance < closest_distance
            closest_distance = distance
            closest_child = child
        end
    end
    return closest_child
end

function split_node!(node::NDTreeNode)


    function find_furthest_point(points::Array{Pareto})
        furthest_point = nothing
        max_distance = -Inf
        for point in points
            distance = mean([euclidean(point.objectives, p.objectives) for p in points if p !== point])
            if distance > max_distance
                max_distance = distance
                furthest_point = point
            end
        end
        return furthest_point
    end

    function create_child_with_point(point::Pareto)
        child = NDTreeNode([deepcopy(point)], copy(point.objectives), copy(point.objectives), NDTreeNode[], node)
        update_ideal_nadir!(child, point)
        return child
    end

    # Keep creating new children until the required number is reached
    while length(node.children) < REQUIRED_CHILDREN_COUNT
        z = find_furthest_point(node.points)
        child = create_child_with_point(z)
        push!(node.children, child)
        setdiff!(node.points, [z])
    end

    # Distribute remaining points among the children
    while !isempty(node.points)
        z = popfirst!(node.points)
        closest_child = find_closest_child(node, z)
        push!(closest_child.points, z)
        update_ideal_nadir!(closest_child, z)
    end
end

# 返回所有点的集合
function postorder_traversal(node::NDTreeNode)
    points = Pareto[]
    if !isempty(node.children)
        for child in node.children
            points = vcat(points, postorder_traversal(child))
        end
    end
    return vcat(points, node.points)
end


function test()
    tree = NDTreeNode(Pareto[], [0, 0], [0, 0], NDTreeNode[], nothing)
    points = [Pareto([1, 1], [0, 0]), Pareto([1, 1], [1, -1])]
    # iteration from 4 to 1
    for i in 10:-1:1
        push!(points, Pareto([1, 1], [2^i, -2^i]))
    end
    for point in points
        update_archive!(tree, point)
    end
    # println(tree)
    paretos = postorder_traversal(tree)
    println(paretos)

end

if abspath(PROGRAM_FILE) == @__FILE__
    test()
end

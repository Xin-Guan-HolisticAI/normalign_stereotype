from normalign_stereotype.core._reference import Reference, cross_product, element_action

def cross_action(A, B, new_axis_name):
    # Validate inputs
    if not isinstance(A, Reference) or not isinstance(B, Reference):
        raise TypeError("Both A and B must be Reference instances")

    # Combine axes from A and B
    combined_axes = list(A.axes)  # Start with axes from A
    for axis in B.axes:
        if axis not in combined_axes:
            combined_axes.append(axis)  # Add axes from B that are not already in A

    # Compute the shape of the resulting tensor
    combined_shape = []
    for axis in combined_axes:
        if axis in A.axes and axis in B.axes:
            # Axes shared by A and B must have the same shape
            if A.shape[A.axes.index(axis)] != B.shape[B.axes.index(axis)]:
                raise ValueError(f"Shape mismatch for shared axis '{axis}': "
                                 f"{A.shape[A.axes.index(axis)]} vs {B.shape[B.axes.index(axis)]}")
            combined_shape.append(A.shape[A.axes.index(axis)])
        elif axis in A.axes:
            # Axis only in A
            combined_shape.append(A.shape[A.axes.index(axis)])
        else:
            # Axis only in B
            combined_shape.append(B.shape[B.axes.index(axis)])

    # Build the new data structure
    def build_data(current_axes, index_dict):
        if not current_axes:
            # Retrieve the function from A and the input from B
            a_indices = {axis: index_dict[axis] for axis in A.axes}
            b_indices = {axis: index_dict[axis] for axis in B.axes}
            func = A.get(**a_indices)
            if not callable(func):
                raise TypeError(f"Element at {a_indices} in A is not a callable function")
            input_val = B.get(**b_indices)
            result = func(input_val)
            if not isinstance(result, list):
                raise TypeError(f"Function at {a_indices} in A must return a list")
            return result
        else:
            axis = current_axes[0]
            axis_size = combined_shape[len(index_dict)]
            sub_list = []
            for i in range(axis_size):
                new_index_dict = index_dict.copy()
                new_index_dict[axis] = i
                sub_list.append(build_data(current_axes[1:], new_index_dict))
            return sub_list

    new_data = build_data(combined_axes, {})

    # Create the new Reference
    new_axes = combined_axes + [new_axis_name]

    """new code start"""

    retrieved_entry = new_data
    for i in range(len(combined_shape)):
        retrieved_entry = retrieved_entry[0]
    #     print("now retrieve:")
    #     print(retrieved_entry)
    # print(len(retrieved_entry))
    new_shape = combined_shape + [len(retrieved_entry) if retrieved_entry else 0]  # New axis size

    """new code end"""

    #delete code
    #new_shape = combined_shape + [len(new_data[0]) if new_data else 0]  # New axis size
    result_ref = Reference(new_axes, new_shape, None)
    result_ref._replace_data(new_data)
    return result_ref


# Create reference A with functions
A = Reference(['x'], (1,), None)
# Assign lambda functions to each position
A.set(lambda z: [z, z * 2], x=0)
print("A (1): ", A.get())

# Create reference B with values
B = Reference(['y', 'z'], (4, 3), None)
# Assign values to each position
B.set(5, y=0, z=0)
B.set(3, y=0, z=1)
B.set(2, y=0, z=2)
B.set(-2, y=1, z=0)
B.set(4, y=1, z=1)
B.set(1, y=1, z=2)
B.set(7, y=2, z=0)
B.set(-3, y=2, z=1)
B.set(0, y=2, z=2)
B.set(4, y=3, z=0)
B.set(9, y=3, z=1)
B.set(-5, y=3, z=2)

print("B (4x3): ", B.get())

# Perform cross-action
result = cross_action(A, B, "operation")

print("Resulting axes:", result.axes)
print("Resulting shape:", result.shape)
print("Resulting data:")
print(result.tensor)

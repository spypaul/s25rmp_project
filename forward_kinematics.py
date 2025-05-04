# Implement the stubs where requested in the comments below
# No imports allowed other than these, do not change these lines
import matplotlib.pyplot as pt
import torch as tr
tr.set_default_dtype(tr.float64) # makes round-off errors much smaller
import rotations as rt
# import ergo_jr

def get_frames(joint_info, joint_angles):
    """
    forward kinematics, joint_angles can be batched (b is batch index)
    joint_angles and outputs are torch tensors
    input:
        joint_info has the same format as in ergo_jr.py
        joint_angles[b,j] is jth joint angle in bth batch sample
    returns (locations, orientations) where
        locations[b,i,:] is 3d location of ith joint in base coordinate frame
        orientations[b,i,:] is 4d orientation versor of ith joint in base coordinate frame
    if leading batch dimension is omitted from input, it will also be omitted in output
    i ranges over all "joints" in joint_info
    j ranges over the non-fixed joints only (where axes in joint_info is not None)
    """
    unbatched = True if joint_angles.dim() == 1 else 0
    if not tr.is_tensor(joint_angles): joint_angles = tr.tensor(joint_angles)
    
    if joint_angles.dim() == 1: joint_angles = joint_angles.unsqueeze(0)

    locations = tr.full((joint_angles.shape[0], len(joint_info)+1,3), 0.)
    orientations = tr.full((joint_angles.shape[0], len(joint_info)+1,4), 0.)
    locations[:, -1, :3] = tr.tensor((0.0, 0.0, 0.05715))
    orientations[:, -1, :4] = tr.tensor((1.0, 0.0, 0.0, 0.0))

    b = joint_angles.shape[0]
    for i, (name, pindex, t, ori, axis) in enumerate(joint_info):
        ap = None
        if pindex == -1: 
            ap = tr.tensor(axis)
        else: 
            ap = tr.tensor(joint_info[pindex][4])
        ap = ap.unsqueeze(0).repeat(b, 1)
        
        if pindex == 6:
            angle = joint_angles[:, 5].unsqueeze(-1)
        else:
            angle = joint_angles[:, pindex].unsqueeze(-1) if pindex != -1 else tr.full((b, 1), 0.)
     
        rp = rt.versor_from(ap, angle)

        qrp = rt.multiply_fastest(orientations[:, pindex, :], rp)
        t = tr.tensor(t).unsqueeze(0).repeat(b, 1)
        locations[:, i, :] = locations[:, pindex, :] + rt.rotate(qrp, t)

        ori = tr.tensor(ori).unsqueeze(0).repeat(b, 1)
        orientations[:, i, :] = rt.multiply_fastest(qrp, ori)
    locations = locations[:,:-1, :]
    orientations = orientations[:,:-1,:]
    if unbatched: 
        locations = locations.squeeze(0)
        orientations = orientations.squeeze(0)


    return (locations, orientations)


    raise NotImplementedError()

def get_jacobian(joint_info, locations, orientations):
    """
    derivatives of joint locations with respect to joint angles
    joint_info, locations and orientations have the format as in get_frames
    output is jacobian, a torch tensor where
        jacobian[b,i,c,j] is d locations[b,i,c] / d joint_angles[b,j]
        b ranges over the batch dimension
        i ranges over all joints, both fixed and non-fixed
        c ranges over their 3d location coordinates
        j ranges over the non-fixed joints only (where axes in joint_info is not None)
    if leading batch dimension is omitted from input, it will also be omitted in output
    you are meant to compute the jacobian from its formula, without using torch autograd
    """

    unbatched = True if locations.dim() == 2 else 0

    if unbatched: 
        locations = locations.unsqueeze(0)
        orientations = orientations.unsqueeze(0)

    b = locations.shape[0]
    jac = tr.full((b, len(joint_info),3, 6), 0.)

    axes = tr.stack([ 
        tr.tensor(axis) 
        for (name, pindex, t, ori, axis) in joint_info if axis is not None
    ]).unsqueeze(0).repeat(b, 1, 1)
    keep_indices = [0, 1, 2, 3, 4, 6]
    world_orientation = orientations[:, keep_indices, :]
    axes = rt.rotate(world_orientation, axes)


    for i in range(0,len(joint_info)):
        for j in keep_indices:
            if i <= j: continue

            if  j == 6:
                jac[:, i, :, 5] = tr.linalg.cross(axes[:,5,:], locations[:, i, :] - locations[:, j, :])
            else:
                jac[:, i, :, j] = tr.linalg.cross(axes[:,j,:], locations[:, i, :] - locations[:, j, :])

    if unbatched:
        jac = jac.squeeze(0)
    return jac
    raise NotImplementedError()


if __name__ == "__main__":

    angles = tr.full((6,), +.2)
    locations, orientations = get_frames(ergo_jr.joint_info, angles)
    jacobian = get_jacobian(ergo_jr.joint_info, locations, orientations)

    jacobian_t = tr.autograd.functional.jacobian(lambda a: get_frames(ergo_jr.joint_info, a)[0], angles)

    print(locations.numpy().round(4))
    print()
    print(orientations.numpy().round(4))
    print()

    print(jacobian.numpy().round(4))
    print()
    print(jacobian_t.numpy().round(4))
    print()

    # ax = pt.gcf().add_subplot(projection='3d')
    # ax.plot(*locations.numpy().T, 'ko-')
    # pt.axis("equal")
    # pt.show()

    assert tr.allclose(jacobian, jacobian_t)



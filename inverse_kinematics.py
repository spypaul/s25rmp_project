# Implement the stubs where requested in the comments below
# No imports allowed other than these, do not change these lines
import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as pt
import torch as tr
tr.set_default_dtype(tr.float64) # makes round-off errors much smaller
import forward_kinematics as fk
import rotations as rt
# import ergo_jr

def location_constraint_fun(joint_info, i, target_t, angles_t):
    """
    Constraint function for a joint location to coincide with a target
    Input:
        joint_info: same format as fk.get_frames
        i: the index of the joint to be constrained
        target_t: shape (3,) tensor with the target location for the joint
        angles_t: shape (J,) tensor with the candidate joint angles (no batching)
    Output:
        sq_dist: shape (1,) tensor of the squared distance from the joint location to the target
    The joint location should be determined by fk.  Batches are not supported.
    J is the number of non-fixed joints only (where axes in joint_info is not None)
    """
    locations, orientations = fk.get_frames(joint_info, angles_t)

    return (( locations[i, :] - target_t)**2).sum().unsqueeze(0)  


    raise NotImplementedError()

def orientation_constraint_fun(joint_info, i, target_t, angles_t):
    """
    Constraint function for a joint location to coincide with a target
    Input:
        joint_info: same format as fk.get_frames
        i: the index of the joint to be constrained
        target_t: shape (3,) tensor with the target location for the joint
        angles_t: shape (J,) tensor with the candidate joint angles (no batching)
    Output:
        sq_dist: shape (1,) tensor of the squared distance from the joint location to the target
    The joint location should be determined by fk.  Batches are not supported.
    J is the number of non-fixed joints only (where axes in joint_info is not None)
    """
    locations, orientations = fk.get_frames(joint_info, angles_t)
    # tvx = rt.rotate(target_t, tr.tensor((1.,0,0)))
    # calcvx = rt.rotate(orientations[i, :], tr.tensor((1.,0,0)))
    # tvy = rt.rotate(target_t, tr.tensor((0,1.,0)))
    # calcvy = rt.rotate(orientations[i, :], tr.tensor((0,1.,0)))
    #
    # return((tr.abs(tvx) - tr.abs(calcvx))**2).sum() + ((tr.abs(tvy) - tr.abs(calcvy))**2).sum()



    return (( orientations[i, :] - target_t)**2).sum().unsqueeze(0)


    raise NotImplementedError()

def location_constraint_jac(joint_info, i, target_t, angles_t):
    """
    Constraint function jacobian for the location constraint
    Input: same format as location_constraint_fun
    Output: jac, a shape (1, J) tensor of the jacobian of the constraint
    """
    locations, orientations = fk.get_frames(joint_info, angles_t)
    fkjac = fk.get_jacobian(joint_info, locations, orientations)

    front = 2*( locations[i, :] - target_t) #should have 2*
    cjac = front.unsqueeze(0) @fkjac[i]
    return cjac

    raise NotImplementedError()

def orientation_constraint_jac(joint_info, i, target_t, angles_t):
    """
    Constraint function jacobian for the location constraint
    Input: same format as location_constraint_fun
    Output: jac, a shape (1, J) tensor of the jacobian of the constraint
    """
    cjac =  tr.autograd.functional.jacobian(lambda a: orientation_constraint_fun(joint_info, i, target_t, a), angles_t)
    return cjac

    raise NotImplementedError()

def location_constraint(joint_info, i, target_t):
    """
    Wraps the location constraint function and jacobian in an so.NonlinearConstraint object
    """

    # convert to/from numpy
    def fun(angles_n):
        angles_t = tr.tensor(angles_n)
        sq_dist_t = location_constraint_fun(joint_info, i, target_t, angles_t)
        return sq_dist_t.numpy()
    def jac(angles_n):
        angles_t = tr.tensor(angles_n)
        jac_t = location_constraint_jac(joint_info, i, target_t, angles_t)
        return jac_t.numpy()

    lo = hi = np.zeros(1) # constraint function should be 0 when satisfied
    return so.NonlinearConstraint(fun, lo, hi, jac)

def orientation_constraint(joint_info, i, target_t):
    """
    Wraps the location constraint function and jacobian in an so.NonlinearConstraint object
    """

    # convert to/from numpy
    def fun(angles_n):
        angles_t = tr.tensor(angles_n)
        sq_dist_t = orientation_constraint_fun(joint_info, i, target_t, angles_t)
        return sq_dist_t.numpy()
    def jac(angles_n):
        angles_t = tr.tensor(angles_n)
        jac_t = orientation_constraint_jac(joint_info, i, target_t, angles_t)
        return jac_t.numpy()

    lo = hi = np.zeros(1) # constraint function should be 0 when satisfied
    return so.NonlinearConstraint(fun, lo, hi, jac)

def angle_norm_obj_and_grad(angles_n):
    """
    so.minimize objective function for the squared angle vector norm
    returns the objective value and its gradient
    """
    return (angles_n @ angles_n), 2*angles_n

if __name__ == "__main__":

    target = tr.tensor([.0, -.07, 0])
    angles = tr.zeros(6)
    sq_dist = location_constraint_fun(ergo_jr.joint_info, 5, target, angles)
    jacobian = location_constraint_jac(ergo_jr.joint_info, 5, target, angles)
    print(sq_dist)
    print(jacobian)

    angles = tr.zeros(6, requires_grad= True)
    out = tr.autograd.functional.jacobian(lambda a: location_constraint_fun(ergo_jr.joint_info, 5, target, a), angles)
    print(out)

    target = tr.tensor([.1, .2, .3])
    angles = tr.full((6,), .2)
    sq_dist = location_constraint_fun(ergo_jr.joint_info, 4, target, angles)
    jacobian = location_constraint_jac(ergo_jr.joint_info, 4, target, angles)
    print(sq_dist)
    print(jacobian)

    angles = tr.full((6,), .2, requires_grad= True)
    out = tr.autograd.functional.jacobian(lambda a: location_constraint_fun(ergo_jr.joint_info, 4, target, a), angles)
    print(out)

    # target = tr.tensor([.1, .2, .3])
    # angles = tr.full((6,), .2)
    # print(location_constraint_fun(ergo_jr.joint_info, 4, target, angles))
    # print(location_constraint_jac(ergo_jr.joint_info, 4, target, angles))

    # target_5 = tr.tensor([.0, -.07, 0])
    # target_7 = tr.tensor([-.02, -.07, 0])


    # print(location_constraint_fun(ergo_jr.joint_info, 5, target_5, tr.zeros(6)))
    # print(location_constraint_jac(ergo_jr.joint_info, 5, target_5, tr.zeros(6)))

    # soln = so.minimize(
    #     angle_norm_obj_and_grad,
    #     x0 = np.zeros(6),
    #     jac = True,
    #     bounds = [(-np.pi, np.pi)] * 6,
    #     constraints = [
    #         location_constraint(ergo_jr.joint_info, 5, target_5),
    #         location_constraint(ergo_jr.joint_info, 7, target_7),
    #     ],
    #     options={'maxiter': 200},
    # )

    # print(soln.message)

    # angles_t = tr.tensor(soln.x)
    # print(soln.x)
    # locations, orientations = fk.get_frames(ergo_jr.joint_info, angles_t)

    # ax = pt.gcf().add_subplot(projection='3d')
    # ax.plot(*locations.numpy().T, 'ko-')
    # ax.plot(*target_5.numpy(), 'ro')
    # ax.plot(*target_7.numpy(), 'bo')
    # pt.axis("equal")
    # pt.show()



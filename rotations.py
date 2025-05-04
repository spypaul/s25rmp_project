# Implement the stubs where requested in the comments below
# No imports allowed other than these, do not change these lines
import torch as tr
tr.set_default_dtype(tr.float64) # makes round-off errors much smaller

# This one is implemented for you
def identity_versor(batch_dims=()):
    """
    Returns the identity versor representing no rotation
    If requested, duplicates the versor across any leading batch dimensions
    batch_dims[k] is the size of the kth batch dimension
    if batch_dims is an int, it is the size of a single leading batch dimension
    """
    if type(batch_dims) == int: batch_dims = (batch_dims,)
    return tr.broadcast_to(tr.tensor([1.,0.,0.,0.]), batch_dims+(4,))

# You need to finish this one
def versor_from(axis, angle, normalize=False):
    """
    Return versor representation of rotation by angle about axis
    axis and angle are tensors with shapes (..., 3) and (..., 1)
    where ... is 0 or more leading batch dimensions
    returned versor batch should have shape (..., 4)
    If normalize is True, each axis must be scaled to unit norm
    axis or angle can be missing batch dimensions in which case they are broadcasted
    angle can also be a float in which case it is promoted to tensor and broadcasted
    """
    if not tr.is_tensor(angle): angle = tr.tensor(angle)
    if angle.dim()<2: angle = angle.view(-1,1)
    if axis.dim()<2: axis = axis.unsqueeze(0)


    if normalize: axis = tr.nn.functional.normalize(axis, dim=-1)
    # x, y, z = axis.unbind(dim=-1)
    x = axis[:, 0]
    y = axis[:, 1]
    z = axis[:, 2]
    angle = angle.squeeze(dim=-1)

    q = tr.stack([tr.cos(angle/2), x*tr.sin(angle/2), y*tr.sin(angle/2), z*tr.sin(angle/2) ], dim = -1)
    return q
    # raise NotImplementedError()

# constant coefficient array for converting versor q to rotation matrix M
# _r[i,j] is a 4x4 coefficient for quadratic function q.T @ _r[i,j] @ q
# Programmatically that means
# M[i,j] = (_r[i,j] * q[None, None, None, :] * q[None, None, :, None]).sum(dim=(-2,-1))
_r = tr.tensor([
    [[[1,0, 0,0],[0,1,0,0],[ 0,0,-1,0],[0,0,0,-1]], [[0,0,0,-1],[0, 0,1,0],[0,1,0,0],[-1,0,0, 0]], [[0, 0,1,0],[ 0, 0,0,1],[1,0, 0,0],[0,1,0,0]]],
    [[[0,0, 0,1],[0,0,1,0],[ 0,1, 0,0],[1,0,0, 0]], [[1,0,0, 0],[0,-1,0,0],[0,0,1,0],[ 0,0,0,-1]], [[0,-1,0,0],[-1, 0,0,0],[0,0, 0,1],[0,0,1,0]]],
    [[[0,0,-1,0],[0,0,0,1],[-1,0, 0,0],[0,1,0, 0]], [[0,1,0, 0],[1, 0,0,0],[0,0,0,1],[ 0,0,1, 0]], [[1, 0,0,0],[ 0,-1,0,0],[0,0,-1,0],[0,0,0,1]]],
]).to(tr.get_default_dtype())

# This one is done for you
def matrix_from(versor):
    """
    Returns the rotation matrix corresponding to the given versor
    versor is shape (..., 4) where ... are 0 or more leading batch dimensions
    returns mats, a batch of rotation matrices with shape (..., 3, 3)
    mats[b,:,k] is rotation versor[b] applied to kth coordinate axis
    """

    # # slow version, based on:
    # # https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    # a, b, c, d = versor.unbind(dim=-1)
    # mats = tr.stack([
    #     tr.stack([a**2 + b**2 - c**2 - d**2, 2*b*c - 2*a*d, 2*b*d + 2*a*c], dim=-1),
    #     tr.stack([2*b*c + 2*a*d, a**2 - b**2 + c**2 - d**2, 2*c*d - 2*a*b], dim=-1),
    #     tr.stack([2*b*d - 2*a*c, 2*c*d + 2*a*b, a**2 - b**2 - c**2 + d**2], dim=-1),
    # ], dim=-2)

    # faster
    mats = (_r * versor[...,None,None,:,None] * versor[...,None,None,None,:]).sum(dim=(-2,-1)) # (..., 3, 3)

    return mats

# You need to do this one
def rotate(q, v):
    """
    Returns v_rot, the result of applying the rotation represented by versor q to the 3D vector v
    q has shape (..., 4) while v and v_rot each have shape (..., 3),
    where ... are 0 or more leading batch dimensions
    q or v can be missing batch dimensions, in which case they are broadcasted
    """
    if q.dim() < 2: q = q.unsqueeze(0)
    if v.dim() < 2: v = v.unsqueeze(0)
    u = q[..., 1:]
    a = q[..., 0:1]
    uxv = tr.cross(u,v, dim = -1)
    av = a*v
    vrot = v + tr.cross(2*u, uxv+av, dim = -1)

    return vrot
    raise NotImplementedError()


# _r2 = tr.tensor([
#     [[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1]],
#     [[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,-1,0]],
#     [[0,0,1,0],[0,0,0,-1],[1,0,0,0],[0,1,0,0]],
#     [[0,0,0,1],[0,0,1,0],[0,-1,0,0],[1,0,0,0]]
# ]).to(tr.get_default_dtype())
# This one is done for you
def multiply(q1, q2, renormalize=True):
    """
    Return the versor q resulting from multiplication of q1 with q2
    q represents a rotation first by q2 and then by q1 (order matters)
    q, q1, and q2 each have shape (..., 4), where ... are 0 or more leading batch dimensions
    if renormalize is True, q should be renormalized to unit length before it is returned
    """
    # slow version, based on:
    # https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    a1, b1, c1, d1 = q1.unbind(dim=-1)
    a2, b2, c2, d2 = q2.unbind(dim=-1)
    q3 = tr.stack([
        a1*a2 - b1*b2 - c1*c2 - d1*d2,
        a1*b2 + b1*a2 + c1*d2 - d1*c2,
        a1*c2 - b1*d2 + c1*a2 + d1*b2,
        a1*d2 + b1*c2 - c1*b2 + d1*a2,
    ], dim=-1)

    if renormalize: q3 = tr.nn.functional.normalize(q3, dim=-1)
    return q3


# Add precomputed global constants here if you want (not required)
_r2 = tr.tensor([
    [[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1]],
    [[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,-1,0]],
    [[0,0,1,0],[0,0,0,-1],[1,0,0,0],[0,1,0,0]],
    [[0,0,0,1],[0,0,1,0],[0,-1,0,0],[1,0,0,0]]
]).to(tr.get_default_dtype())

# Implement a fast quaternion multiplication here
# Semantics are the same as rotations.multiply
def multiply_fastest(q1, q2, renormalize=True):
    q3 = (_r2*q1[...,None,:,None]*q2[...,None,None,:]).sum(dim=(-2,-1))
    if renormalize: q3 = tr.nn.functional.normalize(q3, dim=-1)
    return q3

if __name__ == "__main__":

    # you can edit this area for informal testing if you want
    axis = tr.tensor([[1., 1., 1.], [2., 2., 3.]])
    angle = tr.tensor([[tr.pi/3],[tr.pi/4]])
    axis = tr.tensor([[[1., 1., 1.], [2., 2., 3.]],[[1., 1., 1.], [2., 2., 3.]]])
    angle = tr.tensor([[[tr.pi/3],[tr.pi/4]],[[tr.pi/3],[tr.pi/4]]])
    # axis = tr.tensor([1., 2., 3.])
    # angle = tr.tensor(tr.pi/3)

    quat = versor_from(axis, angle, normalize=True)
    print(quat)
    print(quat.shape)
    mat = matrix_from(quat)
    print(mat)
    print(mat.shape)
    vec1 = rotate(quat, axis)
    print(axis)
    print(axis.shape)
    vec2 = mat @ axis
    print(axis)
    print(vec1)
    print(vec2)

    q1 = versor_from(tr.tensor([[0, 0, 1.],[1., 1., 1.]]), tr.tensor([[tr.pi/2],[tr.pi/3]]))
    q2 = versor_from(tr.tensor([[0, 1, 1.],[1., 2., 1.]]), tr.tensor([[tr.pi/4],[tr.pi/3]]))
    q3 = multiply(q2, q1)
    # print(q1)
    # q2 = versor_from(tr.tensor([[1, 0, 0.]]), tr.tensor([tr.pi/2]))
    # print(q2)
    # q3 = multiply(q2, q1)
    # print(q3)
    # vec = rotate(q3, tr.tensor([[1, 0, 0.]]))
    # print(vec)





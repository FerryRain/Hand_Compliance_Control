"""Verify sensor wrench axes and action/reaction direction from MuJoCo order."""
import torch
from mjlab.sensor.contact_sensor import ContactData, ContactSensor


def transform(normal, tangent, force, torque=None):
    sensor=object.__new__(ContactSensor)
    data=ContactData(normal=normal,tangent=tangent,force=force,torque=torque)
    return sensor._transform_to_global_frame(data)


def test_pure_normal_force_aligns_with_contact_normal():
    normal=torch.tensor([[[0.,0.,1.],[1.,0.,0.]]])
    tangent=torch.tensor([[[1.,0.,0.],[0.,1.,0.]]])
    force=torch.tensor([[[12.,0.,0.],[8.,0.,0.]]])
    output=transform(normal,tangent,force)
    torch.testing.assert_close(output.force,normal*force[...,:1])


def test_secondary_sensor_has_opposite_world_wrench():
    normal=torch.tensor([[[0.,1.,0.]]]);tangent=torch.tensor([[[0.,0.,1.]]])
    force=torch.tensor([[[10.,2.,3.]]]);torque=torch.tensor([[[.4,.2,.1]]])
    a=transform(normal,tangent,force,torque)
    sign=torch.tensor([1.,1.,-1.])
    b=transform(-normal,-tangent,force*sign,torque*sign)
    torch.testing.assert_close(b.force,-a.force)
    torch.testing.assert_close(b.torque,-a.torque)
    torch.testing.assert_close(a.force.norm(dim=-1),force.norm(dim=-1))

import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom

def save_identified_params_to_urdf(
    original_urdf_path,
    output_urdf_path,
    phi_robot,          # Your identified vector (numpy array)
    joint_names,        # List of joint names corresponding to phi_robot order
    armature_vals,      # List of scalar armatures (must match joint_names order)
    save_offsets=True   # If True, saves joint offsets to <calibration> tag
):
    """
    Updates a URDF with identified Link Inertia, Friction, and Manufacturer Armature.
    """
    
    print(f"[URDF] Loading {original_urdf_path}...")
    tree = ET.parse(original_urdf_path)
    root = tree.getroot()
    
    # Helper to format numbers cleanly (avoid 1.23e-19, use 0.0)
    def fmt(val):
        return f"{val:.9f}".rstrip('0').rstrip('.') if abs(val) > 1e-9 else "0.0"

    # --- 1. Iterate through Identifiable Joints ---
    for i, j_name in enumerate(joint_names):
        print(f"[URDF] Updating Joint: {j_name}")
        
        # Find the joint element
        joint = None
        for j in root.findall('joint'):
            if j.get('name') == j_name:
                joint = j
                break
        
        if joint is None:
            print(f"  [Warning] Joint {j_name} not found in URDF. Skipping.")
            continue
            
        # --- A. Extract Parameters for this Joint ---
        # Assuming 13-param standard vector: [m, hx, hy, hz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz, Fv, Fs, Off]
        base = i * 13
        m   = phi_robot[base + 0]
        h   = phi_robot[base + 1 : base + 4] # [mx, my, mz]
        I_O = np.array([
            [phi_robot[base+4], phi_robot[base+5], phi_robot[base+7]],
            [phi_robot[base+5], phi_robot[base+6], phi_robot[base+8]],
            [phi_robot[base+7], phi_robot[base+8], phi_robot[base+9]]
        ])
        fv  = phi_robot[base + 10]
        fs  = phi_robot[base + 11]
        off = phi_robot[base + 12]
        
        armature = armature_vals[i]

        # --- B. Update Link Inertial (Mass & CoM) ---
        child_link_name = joint.find('child').get('link')
        child_link = None
        for l in root.findall('link'):
            if l.get('name') == child_link_name:
                child_link = l
                break
        
        if child_link:
            inertial = child_link.find('inertial')
            if inertial is None:
                inertial = ET.SubElement(child_link, 'inertial')
            
            # 1. Mass
            mass_elem = inertial.find('mass')
            if mass_elem is None: mass_elem = ET.SubElement(inertial, 'mass')
            mass_elem.set('value', fmt(m))
            
            # 2. Origin (Center of Mass)
            # h = m * c  =>  c = h / m
            if m > 1e-6:
                com = h / m
            else:
                com = np.zeros(3)
                
            origin_elem = inertial.find('origin')
            if origin_elem is None: origin_elem = ET.SubElement(inertial, 'origin')
            origin_elem.set('xyz', f"{fmt(com[0])} {fmt(com[1])} {fmt(com[2])}")
            # Keep existing rotation if present, else 0
            if 'rpy' not in origin_elem.attrib:
                origin_elem.set('rpy', "0 0 0")

            # 3. Inertia Tensor (Parallel Axis Theorem)
            # URDF requires Inertia at CoM. We have Inertia at Origin (I_O).
            # I_com = I_O - m * (skew(c)^T * skew(c))
            # Simpler term: I_com_ij = I_O_ij - m * (delta_ij * |c|^2 - c_i * c_j)
            
            c_sq = np.dot(com, com)
            outer_c = np.outer(com, com)
            I_com = I_O - m * (c_sq * np.eye(3) - outer_c)
            
            inertia_elem = inertial.find('inertia')
            if inertia_elem is None: inertia_elem = ET.SubElement(inertial, 'inertia')
            
            inertia_elem.set('ixx', fmt(I_com[0,0]))
            inertia_elem.set('ixy', fmt(I_com[0,1]))
            inertia_elem.set('ixz', fmt(I_com[0,2]))
            inertia_elem.set('iyy', fmt(I_com[1,1]))
            inertia_elem.set('iyz', fmt(I_com[1,2]))
            inertia_elem.set('izz', fmt(I_com[2,2]))

        # --- C. Update Friction (<dynamics>) ---
        dynamics = joint.find('dynamics')
        if dynamics is None:
            dynamics = ET.SubElement(joint, 'dynamics')
        
        # Set damping (Viscous) and friction (Coulomb)
        dynamics.set('damping', fmt(fv))
        dynamics.set('friction', fmt(fs))

        # --- D. Update Offsets (<calibration>) ---
        # Standard ROS/URDF way to store encoder offsets
        if save_offsets:
            calib = joint.find('calibration')
            if calib is None:
                calib = ET.SubElement(joint, 'calibration')
            # 'rising' is the standard attribute for reference position
            calib.set('rising', fmt(off))

        # --- E. Update Armature (<transmission>) ---
        # Search for existing transmission for this joint
        trans_found = False
        for trans in root.findall('transmission'):
            # Look for the <joint> tag inside transmission
            t_joint = trans.find('joint')
            if t_joint is not None and t_joint.get('name') == j_name:
                trans_found = True
                actuator = trans.find('actuator')
                if actuator is None:
                    actuator = ET.SubElement(trans, 'actuator')
                    actuator.set('name', f"{j_name}_motor")
                
                # 1. Set Reduction to 1 (since armature is Output Referred)
                mech_red = actuator.find('mechanicalReduction')
                if mech_red is None: mech_red = ET.SubElement(actuator, 'mechanicalReduction')
                mech_red.text = "1"
                
                # 2. Set Motor Inertia
                mot_inert = actuator.find('motorInertia')
                if mot_inert is None: mot_inert = ET.SubElement(actuator, 'motorInertia')
                mot_inert.text = fmt(armature)
                break
        
        # If no transmission exists, create a simple one
        if not trans_found:
            new_trans = ET.SubElement(root, 'transmission')
            new_trans.set('name', f"{j_name}_trans")
            
            type_elem = ET.SubElement(new_trans, 'type')
            type_elem.text = "transmission_interface/SimpleTransmission"
            
            joint_elem = ET.SubElement(new_trans, 'joint')
            joint_elem.set('name', j_name)
            hw_iface = ET.SubElement(joint_elem, 'hardwareInterface')
            hw_iface.text = "hardware_interface/EffortJointInterface"
            
            act_elem = ET.SubElement(new_trans, 'actuator')
            act_elem.set('name', f"{j_name}_motor")
            
            red_elem = ET.SubElement(act_elem, 'mechanicalReduction')
            red_elem.text = "1"
            
            inert_elem = ET.SubElement(act_elem, 'motorInertia')
            inert_elem.text = fmt(armature)

    # --- 3. Save to File ---
    # Use minidom to prettify (optional, standard ElementTree output is ugly)
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    
    # Remove empty lines generated by minidom
    xmlstr = '\n'.join([line for line in xmlstr.split('\n') if line.strip()])
    
    with open(output_urdf_path, "w") as f:
        f.write(xmlstr)
    
    print(f"[Success] Identified URDF saved to: {output_urdf_path}")
import numpy as np

class DepthProcessor:
    def __init__(self,width:int,height:int,hfov:float):
        self.width = width
        self.height = height
        self.hfov = hfov
        self.vfov = DepthProcessor.calc_vfov(HFOV_rad=hfov,width=width,height=height)

    def calc_vfov(HFOV_rad,height,width):
        return 2 * np.arctan((height / width) * np.tan(HFOV_rad / 2))

    def get_vertical_angle(self,y):
        normalized = (self.height/2-y)/(self.height/2)
        return normalized * (self.vfov/2)
    
    def get_horizontal_angle(self,x):
        normalized = (x-(self.width / 2)) / (self.width / 2)
        return normalized * (self.hfov/ 2)
        
    def verticle_forward_clearence(self,depth_frame,top_rows=50):
        if depth_frame is not None:
            top_frame = depth_frame[:top_rows, :]

            row_indices = np.arange(top_rows).reshape(-1, 1)
            col_indices = np.arange(self.width).reshape(1, -1)

            v_angles = self.get_vertical_angle(row_indices)
            h_angles = self.get_horizontal_angle(col_indices)

            vertical_component = top_frame * np.sin(v_angles)
            forward_component = np.abs(top_frame * np.cos(v_angles) * np.cos(h_angles))
            horizontal_component = top_frame * np.cos(v_angles) * np.sin(h_angles)

            verticle_heights = np.stack((horizontal_component, forward_component, vertical_component), axis=-1)
            return verticle_heights.reshape(-1, 3)
        else:
            return None
        
    def rotate_points(self,points,yaw,pitch,roll):
        yaw,pitch,roll = np.radians([yaw,pitch,roll])
        R_z = np.array([
            [np.cos(yaw),-np.sin(yaw),0],
            [np.sin(yaw),np.cos(yaw),0],
            [0,0,1]
        ])
        R_y = np.array([
            [np.cos(pitch),0,np.sin(pitch)],
            [0,1,0],
            [-np.sin(pitch),0,np.cos(pitch)]
        ])
        R_x = np.array([
            [1,0,0],
            [0,np.cos(roll),-np.sin(roll)],
            [0,np.sin(roll),np.cos(roll)]
        ])
        R = R_z @ R_y @ R_x
        rotated_vector = points @ R.T
        return rotated_vector

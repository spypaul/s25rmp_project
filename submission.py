"""
See example.py for examples
Implement Controller class with your motion planner
"""
import numpy as np
import scipy.optimize as so
import torch as tr
import matplotlib.pyplot as pt
import forward_kinematics as fk
import inverse_kinematics as ik
CUBE_SIDE = 0.01905
BASE_Z = 0.05715
class Controller:
    def __init__(self):
        # load any optimized model data here
        pass
    def interpret_tower(self, goal_poses):
        goal_base_blocks = []
        for k, v in goal_poses.items():
            if v[0][2] <= CUBE_SIDE/2:
                goal_base_blocks.append(k)
            print(k)
            print(v)

        towers = []

        for base in goal_base_blocks:
            bpos = goal_poses[base]
            t = [base]
            for k, v in goal_poses.items():
                if k in goal_base_blocks: continue
                if abs(bpos[0][0] - goal_poses[k][0][0]) < 0.001 and abs(bpos[0][1] - goal_poses[k][0][1]) < 0.001:
                    t.append(k)
            towers.append(t)
        print(towers)

        new_towers = []
        for tower in towers:
            cp_tower = [tower[0]]*len(tower)
            for t in tower:
                if goal_poses[t][0][2] <= CUBE_SIDE/2: continue
                idx = int((goal_poses[t][0][2] - CUBE_SIDE/2) / CUBE_SIDE)
                cp_tower[idx+1] = t
            new_towers.append(cp_tower)
        towers = new_towers

        for i in range(len(towers)):
            for j in range(0, len(towers) - i - 1):
                b1 = towers[j][0]
                first_dist = (goal_poses[b1][0][0] - 0)**2 + (goal_poses[b1][0][1] - 0)**2
                b2 = towers[j+1][0]
                second_dist = (goal_poses[b2][0][0] - 0)**2 + (goal_poses[b2][0][1] - 0)**2
                if first_dist < second_dist:
                    towers[j], towers[j+1] = towers[j+1], towers[j]
        print(towers)
        return towers
    def solve_ik(self, env, target_5, target_7):
        joint_info = env.get_joint_info()

        cur_angles = np.zeros(6)
        idx = 0
        a = env.get_current_angles()
        for k, v in a.items():
            cur_angles[idx] = np.deg2rad(v)
            idx += 1

        soln = so.minimize(
            ik.angle_norm_obj_and_grad,
            x0=cur_angles,
            jac=True,
            bounds=[(-np.pi, np.pi)] * 6,
            constraints=[
                ik.location_constraint(joint_info, 5, target_5),
                ik.location_constraint(joint_info, 7, target_7),
            ],
            options={'maxiter': 300},
        )
        # print(soln.message)
        # print(soln.x)
        angles = np.rad2deg(soln.x)
        goto_angles = {}
        for i in range(0, len(angles)):
            label = "m" + str(i + 1)
            goto_angles[label] = angles[i]

        # print(goto_angles)
        return goto_angles, tr.tensor(soln.x)


    def run(self, env, goal_poses):
        # run the controller in the environment to achieve the goal
        joint_info = env.get_joint_info()
        goal_towers = self.interpret_tower(goal_poses)
        temp_locations = {}
        
        for t in goal_towers:
            for b in t:
                randx = np.random.uniform(-6*CUBE_SIDE, 6*CUBE_SIDE)
                randy = np.random.uniform(-7*CUBE_SIDE, -4*CUBE_SIDE)
                temp_locations[b] = (randx, randy, CUBE_SIDE/2)
        
        '''
        count = 0
        temp_x = -4*CUBE_SIDE
        temp_y = -4*CUBE_SIDE
        for t in goal_towers:
            for b in t:
                temp_locations[b] = (temp_x, temp_y, CUBE_SIDE/2)
                count+= 1
                temp_y -=2*CUBE_SIDE
                if count == 4:
                    temp_x = 4 * CUBE_SIDE
                    temp_y = -4 * CUBE_SIDE
                elif count == 8:
                    temp_x = -5.5 * CUBE_SIDE
                    temp_y = -4 * CUBE_SIDE
                elif count == 12:
                    temp_x = 5.5 * CUBE_SIDE
                    temp_y = -4 * CUBE_SIDE
                elif count == 16:
                    temp_x = -7 * CUBE_SIDE
                    temp_y = -4 * CUBE_SIDE
        '''
        
        current_poses = {}
        for k, _ in goal_poses.items():
            current_poses[k] = env.get_block_pose(k)

        current_towers = self.interpret_tower(current_poses)
        #input("Enter to start")
        duration = 1.
        ik_errors = []
        ik_angle_place = []
        for i in range(len(current_towers)-1, -1, -1):
            tower = current_towers[i]
            for j in range(len(tower)-1, -1, -1):
                block_pos = env.get_block_pose(tower[j])
                a = env.get_current_angles()

                #go to block current position
                target_5 = tr.tensor((block_pos[0][0] + CUBE_SIDE, block_pos[0][1] , block_pos[0][2] ))
                target_7 = tr.tensor((block_pos[0][0] - CUBE_SIDE, block_pos[0][1] , block_pos[0][2]))
                goto_angles, solver_solution = self.solve_ik(env, target_5, target_7)
                env.goto_position(goto_angles, duration)
                fk_results, _ = fk.get_frames(joint_info, solver_solution)
                f5 = fk_results[5]
                f7 = fk_results[7]
                error = tr.norm(f5 - tr.tensor((block_pos[0][0] + CUBE_SIDE, block_pos[0][1] , block_pos[0][2] )))
                ik_errors.append(error)
                error = tr.norm(f7 - tr.tensor((block_pos[0][0] - CUBE_SIDE, block_pos[0][1] , block_pos[0][2])))
                ik_errors.append(error)

                target_5 = tr.tensor((block_pos[0][0] + CUBE_SIDE/2*0.84, block_pos[0][1] , block_pos[0][2]))
                target_7 = tr.tensor((block_pos[0][0] - CUBE_SIDE/2*0.84, block_pos[0][1] , block_pos[0][2]))
                temp, _ = self.solve_ik(env, target_5, target_7)
                goto_angles["m6"] = temp["m6"]
                fix_m6 = goto_angles["m6"]
                env.goto_position(goto_angles, duration)
                # env.settle(5.)
                # input("enter")


                target_5 = tr.tensor((block_pos[0][0] + CUBE_SIDE/2*0.84 , block_pos[0][1] , block_pos[0][2]+ 1.2*BASE_Z ))
                target_7 = tr.tensor((block_pos[0][0] - CUBE_SIDE/2*0.84, block_pos[0][1] , block_pos[0][2] +1.2*BASE_Z))
                goto_angles, solver_solution = self.solve_ik(env, target_5, target_7)
                goto_angles["m6"] = fix_m6
                env.goto_position(goto_angles, duration)
                # env.settle(5.)
                temp_x = temp_locations[tower[j]][0]
                temp_y = temp_locations[tower[j]][1]
                target_5 = tr.tensor((temp_x+ CUBE_SIDE/2*0.84 , temp_y, block_pos[0][2] + 1.*BASE_Z))
                target_7 = tr.tensor((temp_x- CUBE_SIDE/2*0.84 , temp_y, block_pos[0][2] + 1.*BASE_Z ))
                goto_angles, solver_solution = self.solve_ik(env, target_5, target_7)
                goto_angles["m6"] = fix_m6
                env.goto_position(goto_angles, duration)


                #put down
                target_5 = tr.tensor((temp_x+ CUBE_SIDE/2*0.84 , temp_y, block_pos[0][2] ))
                target_7 = tr.tensor((temp_x- CUBE_SIDE/2*0.84, temp_y, block_pos[0][2] ))

                goto_angles, solver_solution = self.solve_ik(env, target_5, target_7)
                goto_angles["m6"] = fix_m6
                env.goto_position(goto_angles, duration)

                goto_angles["m6"] += 20
                env.goto_position(goto_angles, duration)
                env.goto_position(a, duration)

        for t in goal_towers:
            for b in t:
                block_pos = env.get_block_pose(b)
                a = env.get_current_angles()

                #go to block current position
                target_5 = tr.tensor((block_pos[0][0] + CUBE_SIDE, block_pos[0][1]- 0.001 , block_pos[0][2] - 0.001))
                target_7 = tr.tensor((block_pos[0][0] - CUBE_SIDE, block_pos[0][1]- 0.001 , block_pos[0][2] - 0.001))
                goto_angles, solver_solution = self.solve_ik(env, target_5, target_7)
                env.goto_position(goto_angles, duration)

                target_5 = tr.tensor((block_pos[0][0] + CUBE_SIDE/2*0.84, block_pos[0][1]- 0.001 , block_pos[0][2]- 0.001))
                target_7 = tr.tensor((block_pos[0][0] - CUBE_SIDE/2*0.84, block_pos[0][1]- 0.001 , block_pos[0][2]- 0.001))
                temp, _ = self.solve_ik(env, target_5, target_7)
                goto_angles["m6"] = temp["m6"]
                fix_m6 = goto_angles["m6"]
                env.goto_position(goto_angles, duration)
                # env.settle(5.)
                # input("enter")


                target_5 = tr.tensor((block_pos[0][0] + CUBE_SIDE/2*0.84 , block_pos[0][1] , block_pos[0][2]+ 1.2*BASE_Z ))
                target_7 = tr.tensor((block_pos[0][0] - CUBE_SIDE/2*0.84, block_pos[0][1] , block_pos[0][2] +1.2*BASE_Z))
                goto_angles, solver_solution = self.solve_ik(env, target_5, target_7)
                goto_angles["m6"] = fix_m6
                env.goto_position(goto_angles, duration)
                # env.settle(5.)
                goal_location = goal_poses[b]
                target_5 = tr.tensor((goal_location[0][0]+ CUBE_SIDE/2*0.84 , goal_location[0][1], goal_location[0][2] + 1.*BASE_Z))
                target_7 = tr.tensor((goal_location[0][0]- CUBE_SIDE/2*0.84 , goal_location[0][1], goal_location[0][2] + 1.*BASE_Z ))
                goto_angles, solver_solution = self.solve_ik(env, target_5, target_7)
                goto_angles["m6"] = fix_m6
                env.goto_position(goto_angles, duration)


                #put down
                target_5 = tr.tensor((goal_location[0][0]+ CUBE_SIDE/2*0.84 , goal_location[0][1], goal_location[0][2] ))
                target_7 = tr.tensor((goal_location[0][0]- CUBE_SIDE/2*0.84, goal_location[0][1], goal_location[0][2] ))
                goto_angles, solver_solution = self.solve_ik(env, target_5, target_7)
                goto_angles["m6"] = fix_m6
                env.goto_position(goto_angles, duration)
                fk_results, _ = fk.get_frames(joint_info, solver_solution)
                f5 = fk_results[5]
                f7 = fk_results[7]
                error = tr.norm(f5 - tr.tensor((goal_location[0][0]+ CUBE_SIDE/2*0.84 , goal_location[0][1], goal_location[0][2] )))
                ik_errors.append(error)
                error = tr.norm(f7 - tr.tensor((goal_location[0][0]- CUBE_SIDE/2*0.84, goal_location[0][1], goal_location[0][2] )))
                ik_errors.append(error)
                current_block_ori = np.array(env.get_block_pose(b)[1])
                goal_ori = np.array(goal_poses[b][1])
                ik_angle_place.append(2*np.arccos(min(1., np.fabs(current_block_ori @ goal_ori))))
                
                goto_angles["m6"] += 20
                env.goto_position(goto_angles, duration)
                env.goto_position(a, duration)


        print(f"IK errors: {tr.mean(tr.stack(ik_errors))}")
        print(f"Place angle difference: {np.mean(ik_angle_place)}")
        pass
if __name__ == "__main__":

    # you can edit this part for informal testing
    pass


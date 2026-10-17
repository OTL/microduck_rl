import sys, numpy as np, mujoco, importlib.util
sys.path.insert(0, __import__("os").path.dirname(__file__))
import feas
from feas import OUT
from PIL import Image
def render_states(qposes, out, cam_az=90, w=320, h=240):
    m = feas.model(); d = mujoco.MjData(m)
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, w); m.vis.global_.offheight = max(m.vis.global_.offheight, h)
    r = mujoco.Renderer(m, h, w)
    cam = mujoco.MjvCamera(); cam.distance = 1.0; cam.elevation = -10; cam.azimuth = cam_az
    imgs = []
    for q in qposes:
        d.qpos[:] = q; mujoco.mj_forward(m, d); cam.lookat[:] = [q[0], q[1], 0.15]
        opt = mujoco.MjvOption(); opt.geomgroup[:] = [1, 1, 1, 1, 0, 0]
        r.update_scene(d, cam, opt); imgs.append(r.render())
    n = len(imgs); cols = min(n, 6); rows = (n + cols - 1) // cols
    canvas = np.full((rows * h, cols * w, 3), 255, np.uint8)
    for i, im in enumerate(imgs): canvas[(i//cols)*h:(i//cols+1)*h, (i%cols)*w:(i%cols+1)*w] = im
    Image.fromarray(canvas).save(out)
if __name__ == "__main__":
    qs = [feas.start_state("prone"), feas.start_state("supine")]
    render_states(qs, OUT + "/start.png")

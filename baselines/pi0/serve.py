"""Portable form of the evaluated pi0 websocket adapter."""
import argparse
import logging
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--base-params', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    sys.path[:0] = [str(root/'pi0_coribench'), str(root/'openpi/src')]
    import numpy as np
    from pi0_config import make_config
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving import websocket_policy_server as ws
    logging.basicConfig(level=logging.INFO)
    policy = create_trained_policy(make_config(args.repo_id, args.run_dir, args.base_params), args.checkpoint)

    class Checked:
        def infer(self, obs):
            result = policy.infer({'state': obs['observation/state'],
                'image': obs['observation/image'], 'wrist_image': obs['observation/wrist_image'],
                'prompt': obs['prompt']})
            action = np.asarray(result['actions'])
            if action.shape != (50, 7) or not np.isfinite(action).all():
                raise ValueError('Expected finite absolute TCP actions[50,7]')
            return result

    class Server(ws.WebsocketPolicyServer):
        async def run(self):
            async with ws._server.serve(self._handler, self._host, self._port,
                    compression=None, max_size=None, process_request=ws._health_check,
                    ping_timeout=600) as server:
                await server.serve_forever()

    Server(Checked(), host=args.host, port=args.port, metadata={
        'model_name': 'pi0_10k', 'action_horizon': 50, 'execution_steps': 50,
        'action_representation': 'absolute_tcp_xyz_rpy_gripper',
        'norm_source': 'checkpoint_assets'}).serve_forever()


if __name__ == '__main__':
    main()

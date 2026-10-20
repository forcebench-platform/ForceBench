"""Serve either evaluated 16-frame temporal interface over websocket."""
import argparse
import logging
import os
from pathlib import Path
import sys

from recipe import METHODS, make_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--lerobot-home', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--base-params', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    os.environ['HF_LEROBOT_HOME'] = str(args.lerobot_home.resolve())
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root/'runtime/src'))
    import numpy as np
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving import websocket_policy_server as ws
    logging.basicConfig(level=logging.INFO)
    config = make_config(args.method, args.repo_id, args.run_dir, args.base_params)
    policy = create_trained_policy(config, args.checkpoint)

    class Checked:
        def infer(self, obs):
            history = np.asarray(obs['force_history'])
            mask = np.asarray(obs['force_history_is_pad'])
            if history.shape != (16, 6) or mask.shape != (16,) or not np.isfinite(history).all():
                raise ValueError('Expected dense force_history[16,6] and padding mask[16]')
            if not np.array_equal(history[-1], np.asarray(obs['eef_wrench'], dtype=np.float32)):
                raise ValueError('Last history frame must equal current wrist wrench')
            result = policy.infer(obs)
            actions = np.asarray(result['actions'])
            if actions.shape != (50, 7) or not np.isfinite(actions).all():
                raise ValueError('Expected finite absolute TCP actions[50,7]')
            return result

    class Server(ws.WebsocketPolicyServer):
        async def run(self):
            async with ws._server.serve(self._handler, self._host, self._port,
                    compression=None, max_size=None, process_request=ws._health_check,
                    ping_timeout=600) as server:
                await server.serve_forever()

    Server(Checked(), host=args.host, port=args.port, metadata={**policy.metadata,
        'model_name': args.method, 'force_history_frames': 16, 'force_history_hz': 15,
        'action_horizon': 50, 'execution_steps': 50}).serve_forever()


if __name__ == '__main__':
    main()

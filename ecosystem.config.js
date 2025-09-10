module.exports = {
  apps: [
    {
      name: "voice.sonyk.io",
      script: "server_run.py",
      interpreter: "./venv/bin/python",   // Use python3 from venv
      exec_mode: "fork",
      instances: 1,
      autorestart: true,
      watch: false
    }
  ]
}

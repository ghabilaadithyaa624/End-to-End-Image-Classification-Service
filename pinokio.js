const path = require('path')

module.exports = {
  version: "2.0",
  title: "Gradio Application",
  description: "Launches a Gradio application running on port 7860",
  icon: "icon.png",
  menu: async (kernel, info) => {
    // Check if the isolated virtual environment folder exists
    let installed = info.exists("env")

    // Check currently running tasks
    let running = {
      install: info.running("install.json"),
      start: info.running("start.json"),
      update: info.running("update.json"),
      reset: info.running("reset.json")
    }

    if (running.install) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Installing...",
        href: "install.json"
      }]
    } else if (running.start) {
      let local = info.local("start.json")
      if (local && local.url) {
        return [{
          default: true,
          icon: "fa-solid fa-rocket",
          text: "Open Web UI",
          href: local.url
        }, {
          icon: "fa-solid fa-terminal",
          text: "Terminal",
          href: "start.json"
        }]
      } else {
        return [{
          default: true,
          icon: "fa-solid fa-terminal",
          text: "Terminal",
          href: "start.json"
        }]
      }
    } else if (installed) {
      return [{
        default: true,
        icon: "fa-solid fa-power-off",
        text: "Start",
        href: "start.json"
      }, {
        icon: "fa-solid fa-arrows-rotate",
        text: "Update",
        href: "update.json"
      }, {
        icon: "fa-solid fa-trash-can",
        text: "Reset",
        href: "reset.json"
      }]
    } else {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Install",
        href: "install.json"
      }]
    }
  }
}

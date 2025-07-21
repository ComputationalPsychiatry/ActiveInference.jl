"""
ActionModels - set save history
"""

using ActionModels

function ActionModels.set_save_history!(agent::Agent, save_history::Bool)
    agent.save_history = save_history
end
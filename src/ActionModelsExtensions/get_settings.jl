"""
This adds the "get_settings" function to the ensamble of other ActionModels package functions. Note that this is not part of action models but is a supplement.

    get_settings(aif::POMDPActiveInference, target_settings::Vector{String})
Retrieves multiple target settings from an AIF agent. 

    get_settings(aif::POMDPActiveInference, target_setting::String)
Retrieves a single target setting from an AIF agent.

    get_settings(aif::POMDPActiveInference)
Retrieves all settings from an AIF agent.

"""

# Retrieves multiple target settings
function get_settings(aif::POMDPActiveInference, target_settings::Vector{String})
    settings = Dict{String, Any}()

    for target_setting in target_settings
        if hasproperty(aif.settings, Symbol(target_setting))
            value = getproperty(aif.settings, Symbol(target_setting))
            settings[target_setting] = value
        else
            throw(ArgumentError("The specified setting '$target_setting' does not exist in settings."))
        end
    end

    return settings
end

# Retrieves a single setting
function get_settings(aif::POMDPActiveInference, target_setting::String)
    # Check if the state exists in the aif state struct
    if hasproperty(aif.settings, Symbol(target_setting))
        setting_value = getproperty(aif.settings, Symbol(target_setting))

        return setting_value
    else
        # If the target setting is not found, throw an ArgumentError
        throw(ArgumentError("The specified setting '$target_setting' does not exist in settings."))
    end
end


# Retrieves all settings 
function get_settings(aif::POMDPActiveInference)
    setting_struct = aif.settings
    settings_dict = Dict{String, Any}()

    for field in fieldnames(typeof(setting_struct))
        value = getfield(setting_struct, field)
        settings_dict[string(field)] = value
    end

    return settings_dict
end
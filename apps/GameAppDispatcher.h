#pragma once

// Shared --game/--table-type argv routing for AlternatingOptimization and
// CompareCountStrategies: picks blackjack vs. Double Down Madness, strips
// game-specific flags that don't apply to the selected game (with a
// warning), and invokes the matching per-game entry point. Header-only since
// each app links it into exactly one translation unit.

#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

using GameAppMain = int (*)(int, char**);

namespace {

enum class GameType {
    BLACKJACK,
    DOUBLE_DOWN_MADNESS,
    SPANISH_21
};

struct Option {
    std::string name;
    bool inlineValue = false;
};

inline GameType parseGameType(std::string_view value) {
    if (value == "blackjack" || value == "classic")
        return GameType::BLACKJACK;
    if (value == "ddm" || value == "double-down-madness" ||
        value == "double_down_madness") {
        return GameType::DOUBLE_DOWN_MADNESS;
    }
    if (value == "spanish21" || value == "spanish-21" ||
        value == "spanish_21" || value == "spanish") {
        return GameType::SPANISH_21;
    }
    throw std::invalid_argument(
        "Unknown game type '" + std::string(value) +
        "'. Expected blackjack, ddm, or spanish21.");
}

inline const char* gameName(GameType game) {
    switch (game) {
        case GameType::BLACKJACK:            return "blackjack";
        case GameType::DOUBLE_DOWN_MADNESS:  return "ddm";
        case GameType::SPANISH_21:           return "spanish21";
    }
    return "blackjack";
}

inline Option splitOption(const std::string& argument) {
    const size_t equals = argument.find('=');
    if (equals == std::string::npos)
        return {argument, false};
    return {argument.substr(0, equals), true};
}

inline bool isGameSelector(const std::string& name) {
    return name == "--game" || name == "--table-type";
}

inline GameType selectGame(int argc, char** argv) {
    GameType selected = GameType::BLACKJACK;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        const Option option = splitOption(argument);
        if (!isGameSelector(option.name))
            continue;

        if (option.inlineValue) {
            selected = parseGameType(
                argument.substr(argument.find('=') + 1));
        } else {
            if (i + 1 >= argc ||
                std::string_view(argv[i + 1]).starts_with("--")) {
                throw std::invalid_argument(option.name + " requires a value");
            }
            selected = parseGameType(argv[++i]);
        }
    }
    return selected;
}

inline bool hasHelp(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument = argv[i];
        if (argument == "--help" || argument == "-h")
            return true;
    }
    return false;
}

inline std::vector<std::string> filterArguments(int argc,
                                                char** argv,
                                                GameType selected) {
    static const std::unordered_set<std::string> blackjackOnly = {
        "--das", "--sas", "--don", "--rsa", "--hsa",
        "--peek", "--surr", "--bj"
    };
    static const std::unordered_set<std::string> ddmOnly = {
        "--version"
    };
    static const std::unordered_set<std::string> spanishOnly = {
        "--redouble", "--ddr"
    };

    std::vector<std::string> filtered;
    filtered.reserve(static_cast<size_t>(argc) + 2);
    filtered.emplace_back(argv[0]);
    filtered.emplace_back("--game");
    filtered.emplace_back(gameName(selected));

    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        const Option option = splitOption(argument);

        if (isGameSelector(option.name)) {
            if (!option.inlineValue)
                ++i;
            continue;
        }

        const bool unsupported =
            (selected != GameType::BLACKJACK && blackjackOnly.contains(option.name)) ||
            (selected != GameType::DOUBLE_DOWN_MADNESS && ddmOnly.contains(option.name)) ||
            (selected != GameType::SPANISH_21 && spanishOnly.contains(option.name));
        if (!unsupported) {
            filtered.push_back(argument);
            continue;
        }

        std::cerr << "WARNING: ignoring option " << option.name
                  << " for game " << gameName(selected) << ".\n";
        if (!option.inlineValue && i + 1 < argc &&
            !std::string_view(argv[i + 1]).starts_with("--")) {
            ++i;
        }
    }
    return filtered;
}

inline int invoke(GameAppMain appMain, std::vector<std::string>& arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments)
        argv.push_back(argument.data());
    return appMain(static_cast<int>(argv.size()), argv.data());
}

} // namespace

inline int dispatchGameApp(int argc,
                          char** argv,
                          const char* appName,
                          GameAppMain blackjackMain,
                          GameAppMain doubleDownMadnessMain,
                          GameAppMain spanish21Main) {
    try {
        const GameType selected = selectGame(argc, argv);
        if (hasHelp(argc, argv)) {
            std::cout << appName << " game selection:\n"
                      << "  --game <blackjack|ddm|spanish21>  Table implementation "
                         "(default: blackjack)\n"
                      << "  --table-type is accepted as an alias for --game.\n"
                      << "  Irrelevant game-specific options are ignored with a warning.\n\n";
        }

        std::vector<std::string> filtered =
            filterArguments(argc, argv, selected);
        switch (selected) {
            case GameType::BLACKJACK:           return invoke(blackjackMain, filtered);
            case GameType::DOUBLE_DOWN_MADNESS:  return invoke(doubleDownMadnessMain, filtered);
            case GameType::SPANISH_21:           return invoke(spanish21Main, filtered);
        }
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n"
                  << "Use --help for usage.\n";
        return 1;
    }
}

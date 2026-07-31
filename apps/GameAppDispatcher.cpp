#include "GameAppDispatcher.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace {

enum class GameType {
    BLACKJACK,
    DOUBLE_DOWN_MADNESS
};

struct Option {
    std::string name;
    bool inlineValue = false;
};

GameType parseGameType(std::string_view value) {
    if (value == "blackjack" || value == "classic")
        return GameType::BLACKJACK;
    if (value == "ddm" || value == "double-down-madness" ||
        value == "double_down_madness") {
        return GameType::DOUBLE_DOWN_MADNESS;
    }
    throw std::invalid_argument(
        "Unknown game type '" + std::string(value) +
        "'. Expected blackjack or ddm.");
}

const char* gameName(GameType game) {
    return game == GameType::BLACKJACK ? "blackjack" : "ddm";
}

Option splitOption(const std::string& argument) {
    const size_t equals = argument.find('=');
    if (equals == std::string::npos)
        return {argument, false};
    return {argument.substr(0, equals), true};
}

bool isGameSelector(const std::string& name) {
    return name == "--game" || name == "--table-type";
}

GameType selectGame(int argc, char** argv) {
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

bool hasHelp(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument = argv[i];
        if (argument == "--help" || argument == "-h")
            return true;
    }
    return false;
}

std::vector<std::string> filterArguments(int argc,
                                         char** argv,
                                         GameType selected) {
    static const std::unordered_set<std::string> blackjackOnly = {
        "--das", "--sas", "--don", "--rsa", "--hsa",
        "--peek", "--surr", "--bj"
    };
    static const std::unordered_set<std::string> ddmOnly = {
        "--version"
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
            (selected == GameType::BLACKJACK && ddmOnly.contains(option.name)) ||
            (selected == GameType::DOUBLE_DOWN_MADNESS &&
             blackjackOnly.contains(option.name));
        if (!unsupported) {
            filtered.push_back(argument);
            continue;
        }

        std::cerr << "WARNING: ignoring "
                  << (selected == GameType::BLACKJACK ? "DDM-only" : "blackjack-only")
                  << " option " << option.name
                  << " for game " << gameName(selected) << ".\n";
        if (!option.inlineValue && i + 1 < argc &&
            !std::string_view(argv[i + 1]).starts_with("--")) {
            ++i;
        }
    }
    return filtered;
}

int invoke(GameAppMain appMain, std::vector<std::string>& arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments)
        argv.push_back(argument.data());
    return appMain(static_cast<int>(argv.size()), argv.data());
}

} // namespace

int dispatchGameApp(int argc,
                    char** argv,
                    const char* appName,
                    GameAppMain blackjackMain,
                    GameAppMain doubleDownMadnessMain) {
    try {
        const GameType selected = selectGame(argc, argv);
        if (hasHelp(argc, argv)) {
            std::cout << appName << " game selection:\n"
                      << "  --game <blackjack|ddm>  Table implementation "
                         "(default: blackjack)\n"
                      << "  --table-type is accepted as an alias for --game.\n"
                      << "  Irrelevant game-specific options are ignored with a warning.\n\n";
        }

        std::vector<std::string> filtered =
            filterArguments(argc, argv, selected);
        return selected == GameType::BLACKJACK
            ? invoke(blackjackMain, filtered)
            : invoke(doubleDownMadnessMain, filtered);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n"
                  << "Use --help for usage.\n";
        return 1;
    }
}

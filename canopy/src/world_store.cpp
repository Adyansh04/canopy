/**
 * @file world_store.cpp
 * @brief YAML and binary persistence of rooms, objects and coverage.
 */

#include "canopy/world_store.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace canopy
{

namespace
{

constexpr std::uint32_t kMagic   = 0x4D573147U;  // "G1WM", from its first home; older saves load.
constexpr std::uint32_t kVersion = 1;            // world.yaml and objects.bin.
// coverage.bin: 2 adds the floor plan and the wall layers, and drops the unread view directions.
constexpr std::uint32_t kCoverageVersion = 2;

template <typename Layer>
void putLayer(std::ostream& out, const Layer& layer)
{
    out.write(
        reinterpret_cast<const char*>(layer.data()),
        static_cast<std::streamsize>(layer.size() * sizeof(typename Layer::value_type)));
}

template <typename Layer>
void getLayer(std::istream& in, Layer& layer, std::uint32_t cells)
{
    layer.resize(cells);
    in.read(
        reinterpret_cast<char*>(layer.data()),
        static_cast<std::streamsize>(layer.size() * sizeof(typename Layer::value_type)));
}

const char* stateName(ObjectState state)
{
    switch (state)
    {
        case ObjectState::kStale:
            return "stale";
        case ObjectState::kRemoved:
            return "removed";
        default:
            return "active";
    }
}

ObjectState stateOf(const std::string& name)
{
    if (name == "stale")
    {
        return ObjectState::kStale;
    }
    return name == "removed" ? ObjectState::kRemoved : ObjectState::kActive;
}

template <typename T>
void put(std::ostream& out, const T& value)
{
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
bool get(std::istream& in, T& value)
{
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    return static_cast<bool>(in);
}

/// Writes through a temporary file and renames it over @p path, so a crash leaves the old file.
std::string writeAtomically(
    const std::filesystem::path& path, const std::string& contents, std::ios::openmode mode)
{
    const std::filesystem::path temporary = path.string() + ".tmp";
    {
        std::ofstream out(temporary, mode | std::ios::trunc);
        if (!out)
        {
            return "cannot write " + temporary.string();
        }
        out << contents;
        if (!out)
        {
            return "failed writing " + temporary.string();
        }
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    return error ? "cannot replace " + path.string() + ": " + error.message() : std::string{};
}

}  // namespace

std::string saveWorld(const std::string& directory, const WorldSnapshot& snapshot)
{
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error)
    {
        return "cannot create " + directory + ": " + error.message();
    }
    const std::filesystem::path root(directory);

    YAML::Emitter yaml;
    yaml << YAML::BeginMap;
    yaml << YAML::Key << "version" << YAML::Value << kVersion;
    yaml << YAML::Key << "grid" << YAML::Value << YAML::Flow << YAML::BeginMap;
    yaml << YAML::Key << "resolution" << YAML::Value << snapshot.geometry.resolution;
    yaml << YAML::Key << "origin_x" << YAML::Value << snapshot.geometry.origin_x;
    yaml << YAML::Key << "origin_y" << YAML::Value << snapshot.geometry.origin_y;
    yaml << YAML::Key << "width" << YAML::Value << snapshot.geometry.width;
    yaml << YAML::Key << "height" << YAML::Value << snapshot.geometry.height;
    yaml << YAML::EndMap;
    yaml << YAML::Key << "next_room" << YAML::Value << snapshot.next_room;
    yaml << YAML::Key << "rooms" << YAML::Value << YAML::BeginSeq;
    for (const RoomRecord& room : snapshot.rooms)
    {
        yaml << YAML::Flow << YAML::BeginMap;
        yaml << YAML::Key << "id" << YAML::Value << room.id;
        yaml << YAML::Key << "name" << YAML::Value << room.name;
        yaml << YAML::Key << "type" << YAML::Value << room.type;
        yaml << YAML::Key << "type_confidence" << YAML::Value << room.type_confidence;
        yaml << YAML::Key << "type_source" << YAML::Value << room.type_source;
        yaml << YAML::Key << "x" << YAML::Value << room.x;
        yaml << YAML::Key << "y" << YAML::Value << room.y;
        yaml << YAML::EndMap;
    }
    yaml << YAML::EndSeq;
    yaml << YAML::Key << "objects" << YAML::Value << YAML::BeginSeq;
    for (const MappedObject& object : snapshot.objects)
    {
        yaml << YAML::BeginMap;
        yaml << YAML::Key << "id" << YAML::Value << object.id;
        yaml << YAML::Key << "label" << YAML::Value << object.label();
        yaml << YAML::Key << "votes" << YAML::Value << YAML::Flow << YAML::BeginMap;
        for (const auto& [label, weight] : object.votes)
        {
            yaml << YAML::Key << label << YAML::Value << weight;
        }
        yaml << YAML::EndMap;
        yaml << YAML::Key << "name" << YAML::Value << object.name;
        yaml << YAML::Key << "caption" << YAML::Value << object.caption;
        yaml << YAML::Key << "centre" << YAML::Value << YAML::Flow << YAML::BeginSeq
             << object.box_centre.x() << object.box_centre.y() << YAML::EndSeq;
        yaml << YAML::Key << "size" << YAML::Value << YAML::Flow << YAML::BeginSeq
             << object.box_size.x() << object.box_size.y() << object.height() << YAML::EndSeq;
        yaml << YAML::Key << "yaw" << YAML::Value << object.box_yaw;
        yaml << YAML::Key << "observations" << YAML::Value << object.observations;
        yaml << YAML::Key << "first_seen" << YAML::Value << object.first_seen;
        yaml << YAML::Key << "last_seen" << YAML::Value << object.last_seen;
        yaml << YAML::Key << "misses" << YAML::Value << object.misses;
        yaml << YAML::Key << "top_seen" << YAML::Value << object.top_seen;
        yaml << YAML::Key << "state" << YAML::Value << stateName(object.state);
        yaml << YAML::Key << "best_view_score" << YAML::Value << object.best_view_score;
        yaml << YAML::EndMap;
    }
    yaml << YAML::EndSeq;
    yaml << YAML::EndMap;

    std::ostringstream objects(std::ios::binary);
    put(objects, kMagic);
    put(objects, kVersion);
    put(objects, static_cast<std::uint32_t>(snapshot.objects.size()));
    for (const MappedObject& object : snapshot.objects)
    {
        put(objects, static_cast<std::int32_t>(object.id));
        put(objects, static_cast<std::uint32_t>(object.voxels.size()));
        objects.write(
            reinterpret_cast<const char*>(object.voxels.data()),
            static_cast<std::streamsize>(object.voxels.size() * sizeof(std::uint64_t)));
        put(objects, static_cast<std::uint32_t>(object.embedding.size()));
        put(objects, static_cast<std::int32_t>(object.embedding_count));
        objects.write(
            reinterpret_cast<const char*>(object.embedding.data()),
            static_cast<std::streamsize>(object.embedding.size() * sizeof(float)));
    }
    std::ostringstream coverage(std::ios::binary);
    put(coverage, kMagic);
    put(coverage, kCoverageVersion);
    put(coverage, static_cast<std::uint32_t>(snapshot.cells.size()));
    for (const auto* layer : { &snapshot.cells,
                               &snapshot.plan,
                               &snapshot.quality,
                               &snapshot.surface_quality,
                               &snapshot.flags })
    {
        if (layer->size() != snapshot.cells.size())
        {
            return "coverage layers differ in size";
        }
        putLayer(coverage, *layer);
    }
    for (const auto* layer : { &snapshot.structure_hits, &snapshot.band_clear })
    {
        if (layer->size() != snapshot.cells.size())
        {
            return "coverage layers differ in size";
        }
        putLayer(coverage, *layer);
    }

    // world.yaml last: it names the objects, so a save cut short leaves the old names over new
    // voxels rather than names with none.
    std::string failure =
        writeAtomically(root / "objects.bin", objects.str(), std::ios::out | std::ios::binary);
    if (failure.empty())
    {
        failure =
            writeAtomically(root / "coverage.bin", coverage.str(), std::ios::out | std::ios::binary);
    }
    if (failure.empty())
    {
        failure =
            writeAtomically(root / "world.yaml", std::string(yaml.c_str()) + "\n", std::ios::out);
    }
    return failure;
}

bool worldFits(
    const WorldSnapshot& snapshot, const cv::Mat& cells, const GridGeometry& geometry,
    double min_agreement)
{
    const GridGeometry& saved = snapshot.geometry;
    if (saved.width != geometry.width || saved.height != geometry.height ||
        std::abs(saved.resolution - geometry.resolution) > 1e-6 ||
        std::abs(saved.origin_x - geometry.origin_x) > 0.01 ||
        std::abs(saved.origin_y - geometry.origin_y) > 0.01 ||
        snapshot.cells.size() != geometry.cellCount() || cells.empty())
    {
        return false;
    }
    const auto* now        = cells.ptr<std::uint8_t>(0);
    const auto  agree_with = [&](const std::vector<std::uint8_t>& saved) {
        std::size_t agree = 0;
        for (std::size_t index = 0; index < saved.size(); ++index)
        {
            agree += static_cast<std::size_t>(saved[index] == now[index]);
        }
        return static_cast<double>(agree) >= min_agreement * static_cast<double>(saved.size());
    };
    return agree_with(snapshot.cells) ||
           (snapshot.plan.size() == snapshot.cells.size() && agree_with(snapshot.plan));
}

std::string saveOccupancy(
    const std::string& directory, const std::vector<std::int8_t>& data, const GridGeometry& geometry)
{
    if (data.size() != geometry.cellCount())
    {
        return "the map's data does not match its size";
    }
    const std::filesystem::path root(directory);
    std::error_code             error;
    std::filesystem::create_directories(root, error);
    if (error)
    {
        return "cannot create " + directory + ": " + error.message();
    }
    // map_saver's trinary encoding: free white, occupied black, unknown the grey it reads back.
    std::string image =
        "P5\n" + std::to_string(geometry.width) + " " + std::to_string(geometry.height) + "\n255\n";
    const std::size_t header = image.size();
    image.resize(header + geometry.cellCount());
    for (int y = 0; y < geometry.height; ++y)
    {
        // PGM rows run from the top, the map's highest y.
        const std::size_t row =
            header + (static_cast<std::size_t>(geometry.height - 1 - y) * geometry.width);
        for (int x = 0; x < geometry.width; ++x)
        {
            const auto cell = static_cast<std::size_t>(geometry.index(x, y));
            // Signed by definition: -1 is unknown.
            const int value = data[cell];  // NOLINT(bugprone-signed-char-misuse)
            image[row + static_cast<std::size_t>(x)] =
                static_cast<char>(value < 0 ? 205 : (value >= 65 ? 0 : (value <= 25 ? 254 : 205)));
        }
    }
    std::string failure =
        writeAtomically(root / "map.pgm", image, std::ios::out | std::ios::binary);
    if (!failure.empty())
    {
        return failure;
    }
    std::ostringstream yaml;
    yaml.precision(9);
    yaml
        << "image: map.pgm\nmode: trinary\nresolution: " << geometry.resolution << "\norigin: ["
        << geometry.origin_x << ", " << geometry.origin_y
        << ", 0]\nnegate: 0\n"
        // 205 reads back as occupancy 0.196; a free threshold above it would serve unknown as free.
        << "occupied_thresh: 0.65\nfree_thresh: 0.196\n";
    return writeAtomically(root / "map.yaml", yaml.str(), std::ios::out);
}

std::optional<WorldSnapshot> loadWorld(const std::string& directory, std::string& error)
{
    const std::filesystem::path root(directory);
    WorldSnapshot               snapshot;
    std::map<int, std::size_t>  slot_of;
    try
    {
        const YAML::Node yaml = YAML::LoadFile((root / "world.yaml").string());
        if (yaml["version"].as<std::uint32_t>(0) != kVersion)
        {
            error = "world.yaml has an unknown version";
            return std::nullopt;
        }
        const YAML::Node grid = yaml["grid"];
        snapshot.geometry     = { grid["resolution"].as<double>(),
                                  grid["origin_x"].as<double>(),
                                  grid["origin_y"].as<double>(),
                                  grid["width"].as<int>(),
                                  grid["height"].as<int>() };
        snapshot.next_room    = yaml["next_room"].as<int>(1);
        for (const YAML::Node& room : yaml["rooms"])
        {
            snapshot.rooms.push_back({ room["id"].as<std::string>(),
                                       room["name"].as<std::string>(""),
                                       room["type"].as<std::string>(""),
                                       room["type_confidence"].as<double>(0.0),
                                       room["type_source"].as<std::string>(""),
                                       room["x"].as<double>(0.0),
                                       room["y"].as<double>(0.0) });
        }
        for (const YAML::Node& node : yaml["objects"])
        {
            MappedObject object;
            object.id = node["id"].as<int>();
            for (const auto& vote : node["votes"])
            {
                object.votes[vote.first.as<std::string>()] = vote.second.as<float>();
            }
            object.name            = node["name"].as<std::string>("");
            object.caption         = node["caption"].as<std::string>("");
            object.observations    = node["observations"].as<int>(0);
            object.first_seen      = node["first_seen"].as<double>(0.0);
            object.last_seen       = node["last_seen"].as<double>(0.0);
            object.misses          = node["misses"].as<int>(0);
            object.top_seen        = node["top_seen"].as<bool>(false);
            object.state           = stateOf(node["state"].as<std::string>("active"));
            object.best_view_score = node["best_view_score"].as<double>(0.0);
            slot_of[object.id]     = snapshot.objects.size();
            snapshot.objects.push_back(std::move(object));
        }
    }
    catch (const YAML::Exception& exception)
    {
        error = std::string("world.yaml: ") + exception.what();
        return std::nullopt;
    }

    std::ifstream objects(root / "objects.bin", std::ios::binary);
    std::uint32_t magic   = 0;
    std::uint32_t version = 0;
    std::uint32_t count   = 0;
    if (!get(objects, magic) || !get(objects, version) || !get(objects, count) || magic != kMagic ||
        version != kVersion)
    {
        error = "objects.bin is missing or not a world file";
        return std::nullopt;
    }
    for (std::uint32_t i = 0; i < count; ++i)
    {
        std::int32_t  id         = 0;
        std::uint32_t voxels     = 0;
        std::uint32_t dimensions = 0;
        std::int32_t  embedded   = 0;
        if (!get(objects, id) || !get(objects, voxels))
        {
            error = "objects.bin is truncated";
            return std::nullopt;
        }
        std::vector<std::uint64_t> keys(voxels);
        objects.read(
            reinterpret_cast<char*>(keys.data()),
            static_cast<std::streamsize>(voxels * sizeof(std::uint64_t)));
        if (!get(objects, dimensions) || !get(objects, embedded))
        {
            error = "objects.bin is truncated";
            return std::nullopt;
        }
        std::vector<float> embedding(dimensions);
        objects.read(
            reinterpret_cast<char*>(embedding.data()),
            static_cast<std::streamsize>(dimensions * sizeof(float)));
        if (!objects)
        {
            error = "objects.bin is truncated";
            return std::nullopt;
        }
        const auto slot = slot_of.find(id);
        if (slot == slot_of.end())
        {
            continue;
        }
        MappedObject& object   = snapshot.objects[slot->second];
        object.voxels          = std::move(keys);
        object.embedding       = std::move(embedding);
        object.embedding_count = embedded;
    }
    // One named in world.yaml with nothing in objects.bin is nowhere, and never missed.
    std::erase_if(snapshot.objects, [](const MappedObject& object) {
        return object.voxels.empty();
    });

    std::ifstream coverage(root / "coverage.bin", std::ios::binary);
    std::uint32_t cells = 0;
    if (!get(coverage, magic) || !get(coverage, version) || !get(coverage, cells) ||
        magic != kMagic || (version != 1 && version != kCoverageVersion))
    {
        error = "coverage.bin is missing or not a world file";
        return std::nullopt;
    }
    if (version == 1)
    {
        std::vector<std::uint8_t> directions;
        for (auto* layer : { &snapshot.cells,
                             &snapshot.quality,
                             &snapshot.surface_quality,
                             &snapshot.flags,
                             &directions })
        {
            getLayer(coverage, *layer, cells);
        }
    }
    else
    {
        for (auto* layer : { &snapshot.cells,
                             &snapshot.plan,
                             &snapshot.quality,
                             &snapshot.surface_quality,
                             &snapshot.flags })
        {
            getLayer(coverage, *layer, cells);
        }
        getLayer(coverage, snapshot.structure_hits, cells);
        getLayer(coverage, snapshot.band_clear, cells);
    }
    if (!coverage)
    {
        error = "coverage.bin is truncated";
        return std::nullopt;
    }
    return snapshot;
}

std::string setAsideWorld(const std::string& directory, std::string& aside)
{
    const std::filesystem::path root(directory);
    const std::filesystem::path to = root / ("unfit-" + std::to_string(std::time(nullptr)));
    std::error_code             error;
    std::filesystem::create_directories(to, error);
    if (error)
    {
        return "cannot create " + to.string() + ": " + error.message();
    }
    for (const char* name : { "world.yaml",
                              "objects.bin",
                              "coverage.bin",
                              "map.pgm",
                              "map.yaml",
                              "semantic_map.png",
                              "wall_hits.png",
                              "crops" })
    {
        if (std::filesystem::exists(root / name))
        {
            std::filesystem::rename(root / name, to / name, error);
            if (error)
            {
                return "cannot move " + (root / name).string() + ": " + error.message();
            }
        }
    }
    aside = to.string();
    return {};
}

}  // namespace canopy

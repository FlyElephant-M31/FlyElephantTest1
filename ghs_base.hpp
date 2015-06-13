#ifndef GHS_BASE_HPP
#define GHS_BASE_HPP

#include <float.h>

typedef vertex_id_t VertexID;
typedef edge_id_t EdgeID;
typedef weight_t EdgeWeight;
typedef EdgeWeight FragmentID;
typedef unsigned char Level;
typedef unsigned char FindCount;

const EdgeWeight WEIGHT_INFINITY = DBL_MAX;
const FragmentID ISOLATED_VERTEX = WEIGHT_INFINITY;

enum VertexState : unsigned char {
    SLEEPING,
    FIND,
    FOUND,
    HALTED,
};

enum EdgeState : unsigned char {
    BASIC,
    BRANCH,
    REJECTED,
};

enum MessageType : unsigned char {
    SEPARATOR,
    CONNECT,
    INITIATE,
    TEST,
    ACCEPT,
    REJECT,
    CHANGE_ROOT,
    REPORT,
    HALT,
};

class Message {
private:
    FragmentID _fragment;
    MessageType _type;
    Level _level;
    VertexState _state;

    Message(MessageType type);

    static Message const _SEPARATOR, _ACCEPT, _REJECT, _CHANGE_ROOT, _HALT;
public:
    Message();
    static Message connect(Level level);
    static Message initiate(Level level, FragmentID fragment, VertexState state);
    static Message test(Level level, FragmentID fragment);
    static Message report(EdgeWeight weight);
    static const Message &separator();
    static const Message &accept();
    static const Message &reject();
    static const Message &changeRoot();
    static const Message &halt();

    MessageType getType() const;
    Level getLevel() const;
    FragmentID getFragment() const;
    VertexState getState() const;
    EdgeWeight getWeight() const;
};

inline Message::Message() {
}

inline Message::Message(MessageType type) : _type(type) {
}

inline Message Message::connect(Level level) {
    Message message(CONNECT);
    message._level = level;
    return message;
}

inline Message Message::initiate(Level level, FragmentID fragment, VertexState state) {
    Message message(INITIATE);
    message._level = level;
    message._fragment = fragment;
    message._state = state;
    return message;
}

inline Message Message::test(Level level, FragmentID fragment) {
    Message message(TEST);
    message._level = level;
    message._fragment = fragment;
    return message;
}

inline Message Message::report(EdgeWeight weight) {
    Message message(REPORT);
    message._fragment = weight;
    return message;
}

const Message Message::_SEPARATOR(SEPARATOR);
const Message Message::_ACCEPT(ACCEPT);
const Message Message::_REJECT(REJECT);
const Message Message::_CHANGE_ROOT(CHANGE_ROOT);
const Message Message::_HALT(HALT);

inline const Message &Message::separator() {
    return _SEPARATOR;
}

inline const Message &Message::accept() {
    return _ACCEPT;
}

inline const Message &Message::reject() {
    return _REJECT;
}

inline const Message &Message::changeRoot() {
    return _CHANGE_ROOT;
}

inline const Message &Message::halt() {
    return _HALT;
}

inline std::ostream &operator<<(std::ostream &out, VertexState state) {
    switch (state) {
        case SLEEPING:
            return out << "SLEEPING";
        case FIND:
            return out << "FIND";
        case FOUND:
            return out << "FOUND";
        default:
            return out << "UNKNOWN";
    }
}

inline std::ostream &operator<<(std::ostream &out, Level level) {
    return out << (unsigned) level;
}

inline std::ostream &operator<<(std::ostream &out, const Message &message) {
    switch (message.getType()) {
        case SEPARATOR:
            return out << "Separator";
        case CONNECT:
            return out << "Connect(" << message.getLevel() << ")";
        case INITIATE:
            return out << "Initiate(" << message.getLevel() << "," << message.getFragment() << "," << message.getState() << ")";
        case TEST:
            return out << "Test(" << message.getLevel() << "," << message.getFragment() << ")";
        case ACCEPT:
            return out << "Accept";
        case REJECT:
            return out << "Reject";
        case CHANGE_ROOT:
            return out << "Change Root";
        case REPORT:
            return out << "Report(" << message.getWeight() << ")";
        default:
            return out << "Unknown";
    }
}

inline MessageType Message::getType() const {
    return _type;
}

inline Level Message::getLevel() const {
    return _level;
}

inline FragmentID Message::getFragment() const {
    return _fragment;
}

inline FragmentID Message::getWeight() const {
    return _fragment;
}

inline VertexState Message::getState() const {
    return _state;
}

#endif
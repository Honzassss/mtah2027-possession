"""Write the ATD pair: FIFA/EPTS metadata XML and the raw "frame:slots:ball" text file.

Structure follows Metrica's public Sample Game 3 metadata, extended with the speed channel DATA.md
describes. Ball groups also carry a speed field, which kloppy skips because it is not a position
channel; player speed reaches kloppy through the speed sensor.
"""
NAN_GROUP = "NaN,NaN,NaN"


def _provider_parameters(tag, pairs, indent):
    pad = " " * indent
    lines = ["{}<{}>".format(pad, tag)]
    for name, value in pairs:
        lines.append("{}  <ProviderParameter>".format(pad))
        lines.append("{}    <Name>{}</Name>".format(pad, name))
        lines.append("{}    <Value>{}</Value>".format(pad, value) if value != "" else "{}    <Value/>".format(pad))
        lines.append("{}  </ProviderParameter>".format(pad))
    lines.append("{}</{}>".format(pad, tag))
    return lines


def write_metadata(path, timeline, cfg, home_attacks_right_first_half):
    n_home, n_away = cfg["slots"]["home"], cfg["slots"]["away"]
    n_slots = n_home + n_away
    width, height = cfg["pitch"]["metadata_size"]
    blocks = {half: (first, last) for half, first, last in timeline.blocks}
    home_id, away_id = "TWINH", "TWINA"
    home_direction = "left_to_right" if home_attacks_right_first_half else "right_to_left"
    away_direction = "right_to_left" if home_attacks_right_first_half else "left_to_right"

    lines = ['<?xml version="1.0" encoding="utf-8"?>',
             '<main xmlns:tns="FIFADataTransferFormatEPTSNamespace/FIFADataTransferFormatEPTS" '
             'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
             'xsi:schemaLocation="FIFADataTransferFormatEPTSNamespace/FIFADataTransferFormatEPTS '
             'FIFADataTransferFormatEPTS.xsd">',
             "  <Metadata>",
             "    <GlobalConfig>",
             "      <FileDate>2027-01-01T00:00:00.000Z</FileDate>",
             "      <FileName>TWIN_NOT_REAL_DATA_BAR-ATH_tactical_atd</FileName>",
             "      <TrackingType>Optical</TrackingType>",
             "      <ProviderName>Metrica Sports</ProviderName>",
             "      <FrameRate>{}</FrameRate>".format(timeline.fps)]
    # frame_id = video frame + 1
    lines += _provider_parameters("ProviderGlobalParameters", [
        ("first_half_start", blocks[1][0] + 1), ("first_half_end", blocks[1][1] + 1),
        ("second_half_start", blocks[2][0] + 1), ("second_half_end", blocks[2][1] + 1),
        ("first_extra_half_start", ""), ("first_extra_half_end", ""),
        ("second_extra_half_start", ""), ("second_extra_half_end", "")], 6)
    lines += ["    </GlobalConfig>",
              "    <Sessions>",
              '      <Session id="session1">',
              "        <SessionType>Match</SessionType>",
              "        <Start>2027-01-01T00:00:00.000Z</Start>",
              "        <MatchParameters>",
              '          <Score idLocalTeam="{}" idVisitingTeam="{}">'.format(home_id, away_id),
              "            <LocalTeamScore>0</LocalTeamScore>",
              "            <VisitingTeamScore>0</VisitingTeamScore>",
              "          </Score>",
              "        </MatchParameters>",
              "        <FieldSize>",
              "          <Width>{}</Width>".format(width),
              "          <Height>{}</Height>".format(height),
              "        </FieldSize>",
              '        <Competition id="TWIN">Twin, not competition data</Competition>']
    lines += _provider_parameters("ProviderSessionParameters", [("matchday", "")], 8)
    lines += ["      </Session>",
              "    </Sessions>",
              "    <Teams>"]
    for team_id, name, direction in ((home_id, "TWIN-HOME", home_direction), (away_id, "TWIN-AWAY", away_direction)):
        lines.append('      <Team id="{}">'.format(team_id))
        lines.append("        <Name>{}</Name>".format(name))
        lines += _provider_parameters("ProviderTeamsParameters", [("attack_direction_first_half", direction)], 8)
        lines.append("      </Team>")
    lines += ["    </Teams>", "    <Players>"]
    for slot in range(1, n_slots + 1):
        team_id = home_id if slot <= n_home else away_id
        lines.append('      <Player id="{}" teamId="{}">'.format(slot, team_id))
        lines.append("        <Name>Track {}</Name>".format(slot))
        lines.append("        <ShirtNumber>{}</ShirtNumber>".format(slot))
        lines += _provider_parameters("ProviderPlayerParameters", [
            ("position_type", "Unknown"), ("position_index", -1), ("position_x", 0), ("position_y", 0)], 8)
        lines.append("      </Player>")
    lines += ["    </Players>",
              "    <Devices>",
              '      <Device id="device1">',
              "        <Name>Device 1</Name>",
              "        <Sensors>",
              '          <Sensor id="position">',
              "            <Name>Position</Name>",
              "            <Channels>",
              '              <Channel id="x">',
              "                <Name>position_x</Name>",
              "                <Unit>normalized</Unit>",
              "              </Channel>",
              '              <Channel id="y">',
              "                <Name>position_y</Name>",
              "                <Unit>normalized</Unit>",
              "              </Channel>",
              "            </Channels>",
              "          </Sensor>",
              '          <Sensor id="speed">',
              "            <Name>Speed</Name>",
              "            <Channels>",
              '              <Channel id="s">',
              "                <Name>speed</Name>",
              "                <Unit>m/s</Unit>",
              "              </Channel>",
              "            </Channels>",
              "          </Sensor>",
              "        </Sensors>",
              "      </Device>",
              "    </Devices>",
              "    <PlayerChannels>"]
    for slot in range(1, n_slots + 1):
        for channel in "xys":
            lines.append('      <PlayerChannel channelId="{0}" id="player{1}_{0}" playerId="{1}"/>'.format(channel, slot))
    lines += ["    </PlayerChannels>",
              "  </Metadata>",
              "  <DataFormatSpecifications>",
              '    <DataFormatSpecification endFrame="{}" separator=":" startFrame="1">'.format(timeline.n),
              '      <StringRegister name="frameCount"/>',
              '      <SplitRegister separator=";">']
    for slot in range(1, n_slots + 1):
        lines.append('        <SplitRegister separator=",">')
        for channel in "xys":
            lines.append('          <PlayerChannelRef playerChannelId="player{}_{}"/>'.format(slot, channel))
        lines.append("        </SplitRegister>")
    lines += ["      </SplitRegister>",
              '      <SplitRegister separator=",">',
              '        <BallChannelRef channelId="x"/>',
              '        <BallChannelRef channelId="y"/>',
              '        <BallChannelRef channelId="s"/>',
              "      </SplitRegister>",
              "    </DataFormatSpecification>",
              "  </DataFormatSpecifications>",
              "</main>",
              ""]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def write_raw(path, slot_x, slot_y, slot_s, ball_x, ball_y, ball_s):
    """One line per frame, frame_id from 1. Numbers as in Metrica's sample: 5 decimals on positions."""
    n_frames, n_slots = slot_x.shape
    with open(path, "w") as handle:
        for frame in range(n_frames):
            xs = slot_x[frame].tolist()
            ys = slot_y[frame].tolist()
            ss = slot_s[frame].tolist()
            groups = [NAN_GROUP if x != x else "{:.5f},{:.5f},{:.3f}".format(x, y, s)
                      for x, y, s in zip(xs, ys, ss)]
            x = float(ball_x[frame])
            if x != x:
                ball = NAN_GROUP
            else:
                speed = float(ball_s[frame])
                ball = "{:.5f},{:.5f},{}".format(x, float(ball_y[frame]), "NaN" if speed != speed else "{:.3f}".format(speed))
            handle.write("{}:{}:{}\n".format(frame + 1, ";".join(groups), ball))

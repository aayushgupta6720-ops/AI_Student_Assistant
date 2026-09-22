# Project plan: home network upgrade

Goal: replace the ISP router with a proper setup by the end of October.

## Milestones

1. **Week 1** - Order hardware: a mini PC for the firewall, a managed
   8-port switch, and two Wi-Fi access points. Budget: 450 dollars.
2. **Week 2** - Install OPNsense on the mini PC, configure WAN/LAN, and
   confirm the internet works through it before touching anything else.
3. **Week 3** - Set up VLANs: 10 for trusted devices, 20 for IoT, 30 for
   guests. IoT devices must not be able to reach the trusted VLAN.
4. **Week 4** - Move the access points over, set per-VLAN SSIDs, and run
   speed tests from each room.

## Open questions

- Do I need a PoE switch for the access points, or use their injectors?
- Is a 2.5 GbE uplink worth the extra cost right now?

## Risks

The main risk is the family losing internet during the cutover. Do the
cutover on a Saturday morning and keep the ISP router ready to plug back in.

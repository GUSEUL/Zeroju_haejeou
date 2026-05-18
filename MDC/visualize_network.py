import pygame
import numpy as np
import math
from mdc_gym_env import MDCOffloadingEnv
from compare_baselines import train_sarsa, train_q_learning, AllLocalAgent, RandomAgent, ThresholdAgent

# --- Colors & Constants ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (50, 150, 255)
RED = (255, 50, 50)
GREEN = (50, 200, 50)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
DARK_GRAY = (50, 50, 50)

SCREEN_WIDTH = 1100
SCREEN_HEIGHT = 700
CENTER = (SCREEN_WIDTH // 2 + 100, SCREEN_HEIGHT // 2)
NODE_RADIUS = 40
NEIGHBOR_RADIUS = 220

class Visualizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("MDC Offloading: High-Fidelity Visualization")
        self.font = pygame.font.SysFont("Arial", 18)
        self.header_font = pygame.font.SysFont("Arial", 22, bold=True)
        self.clock = pygame.time.Clock()
        
        self.neighbor_pos = []
        for i in range(6):
            angle = math.radians(i * 60 - 90)
            x = CENTER[0] + NEIGHBOR_RADIUS * math.cos(angle)
            y = CENTER[1] + NEIGHBOR_RADIUS * math.sin(angle)
            self.neighbor_pos.append((int(x), int(y)))
            
        self.packets = []

    def draw_node(self, pos, label, color, physical_q=0):
        pygame.draw.circle(self.screen, color, pos, NODE_RADIUS)
        pygame.draw.circle(self.screen, BLACK, pos, NODE_RADIUS, 2)
        text = self.font.render(label, True, BLACK)
        self.screen.blit(text, (pos[0] - text.get_width()//2, pos[1] - NODE_RADIUS - 20))
        
        # Queue Gauge
        bar_width = 70
        bar_height = 12
        bar_x = pos[0] - bar_width // 2
        bar_y = pos[1] + NODE_RADIUS + 5
        pygame.draw.rect(self.screen, GRAY, (bar_x, bar_y, bar_width, bar_height))
        fill_ratio = min(1.0, physical_q / 50.0)
        fill_color = GREEN if fill_ratio < 0.4 else (ORANGE if fill_ratio < 0.8 else RED)
        pygame.draw.rect(self.screen, fill_color, (bar_x, bar_y, int(bar_width * fill_ratio), bar_height))
        pygame.draw.rect(self.screen, BLACK, (bar_x, bar_y, bar_width, bar_height), 1)
        q_text = self.font.render(f"Q: {physical_q}", True, BLACK)
        self.screen.blit(q_text, (pos[0] - q_text.get_width()//2, bar_y + bar_height + 2))

    def add_packet(self, start, end):
        self.packets.append({"start": start, "end": end, "progress": 0.0})

    def update_packets(self):
        new_packets = []
        for p in self.packets:
            p["progress"] += 0.12 
            if p["progress"] < 1.0:
                curr_x = p["start"][0] + (p["end"][0] - p["start"][0]) * p["progress"]
                curr_y = p["start"][1] + (p["end"][1] - p["start"][1]) * p["progress"]
                pygame.draw.circle(self.screen, BLUE, (int(curr_x), int(curr_y)), 7)
                new_packets.append(p)
        self.packets = new_packets

    def run_simulation(self, env, agents_dict):
        running = True
        agent_names = list(agents_dict.keys())
        current_agent_idx = 0
        state, _ = env.reset()
        step_count = 0
        total_energy = 0
        
        while running:
            self.screen.fill(WHITE)
            current_name = agent_names[current_agent_idx]
            agent_data = agents_dict[current_name]
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN:
                    if pygame.K_1 <= event.key <= pygame.K_5:
                        current_agent_idx = event.key - pygame.K_1
                        state, _ = env.reset()
                        step_count, total_energy = 0, 0
                        self.packets = []
            
            # Decision
            action = np.argmax(agent_data[0][env.get_state_index(state)]) if agent_data[1] else agent_data[0].choose_action(state)
            next_state, reward, term, trunc, info = env.step(action)
            total_energy += info['energy']

            # Visualization Logic Refinement
            # 0: Local (Orange), 1-6: Offload (Blue), 7: Drop (Dark Gray/Red)
            if action == 0:
                main_color = ORANGE
                self.add_packet(CENTER, (CENTER[0], CENTER[1]-20)) 
            elif 1 <= action <= 6:
                main_color = BLUE
                self.add_packet(CENTER, self.neighbor_pos[action - 1]) 
            else: # Action 7: Drop
                main_color = DARK_GRAY
            
            if info['is_dropped']: 
                main_color = RED 

            # Draw Central MDC
            self.draw_node(CENTER, "Main MDC", main_color, physical_q=info['q_size'])
            
            # Draw Neighbor Nodes with their own queues
            for i in range(6):
                node_color = YELLOW if (action == i + 1) else GRAY
                # env.neighbor_queues access directly for visualization
                self.draw_node(self.neighbor_pos[i], f"N{i+1}", node_color, physical_q=env.neighbor_queues[i])

            self.update_packets()

            # Sidebar UI
            pygame.draw.rect(self.screen, GRAY, (0, 0, 280, SCREEN_HEIGHT))
            pygame.draw.line(self.screen, BLACK, (280, 0), (280, SCREEN_HEIGHT), 2)
            
            title = self.header_font.render("AGENT SELECTION", True, BLACK)
            self.screen.blit(title, (20, 20))
            for i, name in enumerate(agent_names):
                color = RED if i == current_agent_idx else BLACK
                txt = self.font.render(f"{'> ' if i==current_agent_idx else '  '}{i+1}. {name}", True, color)
                self.screen.blit(txt, (30, 60 + i * 30))

            info_y = 260
            stats_title = self.header_font.render("REAL-TIME STATS", True, BLACK)
            self.screen.blit(stats_title, (20, info_y))
            stats = [
                f"Step: {step_count}",
                f"Action: {action}",
                f"Reward: {reward:.2f}",
                f"Energy (J): {info['energy']:.2f}",
                f"Total Energy: {total_energy:.1f} J",
                f"Queue Level: {state[3]}",
                f"Dropped: {'YES' if info['is_dropped'] else 'NO'}"
            ]
            for i, s in enumerate(stats):
                txt = self.font.render(s, True, BLACK)
                self.screen.blit(txt, (30, info_y + 40 + i * 25))

            pygame.display.flip()
            self.clock.tick(12)
            state = next_state
            step_count += 1
            if term or trunc:
                state, _ = env.reset()
                step_count, total_energy = 0, 0
        pygame.quit()

if __name__ == "__main__":
    env = MDCOffloadingEnv()
    print("Loading/Training agents for visualization...")
    # Use default 5000-episode checkpoints if they exist
    sarsa_q, _ = train_sarsa(env, episodes=5000)
    ql_q, _ = train_q_learning(env, episodes=5000)
    agents = {"SARSA": (sarsa_q, True), "Q-Learning": (ql_q, True), "All-Local": (AllLocalAgent(), False), "Random": (RandomAgent(), False), "Threshold": (ThresholdAgent(), False)}
    Visualizer().run_simulation(env, agents)

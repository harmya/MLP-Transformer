from manim import *

class attention(Scene):
    def construct(self):
        # show text
        sentence = Text("the laughter makes me happy").scale(0.7).to_edge(UP).shift(0.1*DOWN)
        down_arrow = Tex(r"$\downarrow$").scale(2).next_to(sentence, DOWN)
        matrix_elements = [Text("laughter"), Text("and"), Text("joy"), Text("are"), Text("sad")]
        matrix_elements_text = [Tex("$w_1$"), Tex("$w_2$"), Tex("$w_3$"), Tex("$w_4$"), Tex("$w_5$")]
        matrix_elements_text_transpose = list(zip(*matrix_elements_text))
    
        help_text = Tex("where $w_n$ is a word embedding of dimension $d$").scale(0.7).to_edge(DOWN).shift(0.1*UP).to_edge(LEFT).shift(0.1*RIGHT)
        matrix1 = MobjectMatrix([[matrix_elements[0]], [matrix_elements[1]], [matrix_elements[2]], [matrix_elements[3]], [matrix_elements[4]]], element_alignment_corner=DOWN).scale(0.7).next_to(down_arrow, DOWN)
        matrix2 = MobjectMatrix([[matrix_elements_text[0]], [matrix_elements_text[1]], [matrix_elements_text[2]], [matrix_elements_text[3]], [matrix_elements_text[4]]]).scale(0.7).next_to(down_arrow, DOWN)
        
        equals_matrix = Tex("=").next_to(matrix1, RIGHT).shift(0.5*LEFT)
        Q = Text("Q", color=RED).scale(0.7).next_to(equals_matrix, RIGHT)
        equals_Q = Tex("=").next_to(Q, RIGHT)
        K = Text("K", color=GREEN).scale(0.7).next_to(equals_Q, RIGHT)

        self.play(Write(sentence))
        self.wait(0.5)
        self.play(Write(down_arrow))
        self.wait(0.5)
        self.play(Write(matrix1))
        self.wait(0.5)
        self.play(Transform(matrix1, matrix2))
        self.wait(0.5)
        self.play(Write(help_text))
        self.wait(0.5)
        self.play(Write(equals_matrix))
        self.wait(0.5)
        self.play(Write(Q))
        self.wait(0.5)
        self.play(Write(equals_Q))
        self.wait(0.5)
        self.play(Write(K))
        self.wait(0.5)
        
        # clear screen
        self.play(FadeOut(sentence), FadeOut(down_arrow), FadeOut(matrix1), FadeOut(help_text), FadeOut(equals_matrix), FadeOut(Q), FadeOut(equals_Q), FadeOut(K))
        self.wait(0.5)

        matrix_q = matrix2.copy().to_edge(LEFT).shift(0.7*RIGHT)
        dim_q = Tex("$5 \\times d$").next_to(matrix_q, DOWN).scale(0.7)
        Q.next_to(matrix_q, UP)
        matrix_q_group = VGroup(matrix_q, dim_q, Q)
        

        matrix_k = matrix2.copy().next_to(matrix_q, RIGHT)
        dim_k = Tex("$5 \\times d$").next_to(matrix_k, DOWN).scale(0.7)
        K.next_to(matrix_k, UP)
        matrix_k_group = VGroup(matrix_k, dim_k, K)
        

        matrix_k_t = MobjectMatrix(matrix_elements_text_transpose, h_buff=0.5).move_to(matrix_k).shift(1*RIGHT)
        dim_k_transpose = Tex("$d \\times 5$").next_to(matrix_k_t, DOWN).scale(0.7)
        k_transpose = Tex("$K^T$", color=GREEN).next_to(matrix_k_t, UP).scale(0.7)
        matrix_k_transpose_group = VGroup(matrix_k_t, dim_k_transpose, k_transpose)

        qkt_matrix = [
            [Tex("$w_1 \\cdot w_1$"), Tex("$w_1 \\cdot w_2$"), Tex("$w_1 \\cdot w_3$"), Tex("$w_1 \\cdot w_4$"), Tex("$w_1 \\cdot w_5$")],
            [Tex("$w_2 \\cdot w_1$"), Tex("$w_2 \\cdot w_2$"), Tex("$w_2 \\cdot w_3$"), Tex("$w_2 \\cdot w_4$"), Tex("$w_2 \\cdot w_5$")],
            [Tex("$w_3 \\cdot w_1$"), Tex("$w_3 \\cdot w_2$"), Tex("$w_3 \\cdot w_3$"), Tex("$w_3 \\cdot w_4$"), Tex("$w_3 \\cdot w_5$")],
            [Tex("$w_4 \\cdot w_1$"), Tex("$w_4 \\cdot w_2$"), Tex("$w_4 \\cdot w_3$"), Tex("$w_4 \\cdot w_4$"), Tex("$w_4 \\cdot w_5$")],
            [Tex("$w_5 \\cdot w_1$"), Tex("$w_5 \\cdot w_2$"), Tex("$w_5 \\cdot w_3$"), Tex("$w_5 \\cdot w_4$"), Tex("$w_5 \\cdot w_5$")]
        ]

        equals_qkt = Tex("=").next_to(matrix_k_transpose_group, RIGHT)
        qkt_matrix = MobjectMatrix(qkt_matrix, h_buff=2).scale(0.7).next_to(equals_qkt, RIGHT)
        qkt = Tex("$Q \\cdot K^T$", color=ORANGE).next_to(qkt_matrix, UP).scale(0.7)
        qkt_dim = Tex("$5 \\times 5$").next_to(qkt_matrix, DOWN).scale(0.7)
        qkt_group = VGroup(qkt_matrix, qkt, qkt_dim)

        self.play(Write(matrix_q_group))
        self.wait(0.5)
        self.play(Write(matrix_k_group))
        self.wait(0.5)
        self.play(Transform(matrix_k_group, matrix_k_transpose_group))
        self.wait(0.5)
        self.play(Write(equals_qkt))
        self.wait(0.5)
        self.play(Write(qkt_group))
        self.wait(0.5)


        



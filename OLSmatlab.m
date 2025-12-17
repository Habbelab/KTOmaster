rng(1000);
data = readtable('TR indekser.xlsx');
data = flipud(data);  % Ældste først

% Fjern rækker med NaN i relevante kolonner
relevanteKolonner = {'dlnp', 'x0HvisMeanReverse', 'regime_3', 'regime_5', 'regime_7'};
data = data(~any(ismissing(data(:, relevanteKolonner)), 2), :);

% Opret dummy-variable for regimer
regime0 = data.x0HvisMeanReverse == 0;
regime1 = data.x0HvisMeanReverse == 1;
% Opret lagget dLnP korrekt (efter vendt rækkefølge)
data.lag_dlnp = [NaN; data.dlnp(1:end-1)];
% Fjern første observation (lag = NaN)
data = data(2:end, :);
regime0 = regime0(2:end);
regime1 = regime1(2:end);

% Definér regressorer
lag_r0 = data.lag_dlnp .* regime0;
lag_r1 = data.lag_dlnp .* regime1;
cons_r1 = regime0;
cons_r2 = regime1;


% Saml X og y
X = [cons_r1, cons_r2, lag_r0, lag_r1];
y = data.dlnp;
dato = data.Dato;

% Plot tidsserie
figure
plot(dato, y)
ylabel('Rate (%)')
title ('Monthly changes in prices')

% Estimér OLS
olsmdl = fitlm(X, y, "Intercept",false);
disp(olsmdl)


% Initiale submodels til Markov
sigma2 = olsmdl.MSE;
cons1 = olsmdl.Coefficients.Estimate(1);
cons2 = olsmdl.Coefficients.Estimate(2);
AR1 = olsmdl.Coefficients.Estimate(3);
AR2 = olsmdl.Coefficients.Estimate(4);

MSmdl1_ini = arima('Constant', cons1, 'AR', AR1, 'Variance', sigma2);
MSmdl2_ini = arima('Constant', cons2, 'AR', AR2, 'Variance', sigma2);

% Initial switching mekanisme [p_11, p_12; p_21, p_22]
regimeNavne = ["Regime 1 - Mean reversion", "Regime 2 - Trend following"];
P = [0.51, 0.49; 0.83, 0.17];
mc_ini = dtmc(P, 'StateNames',regimeNavne);

% Switching model
MSmdl3_ini = msVAR(mc_ini,[MSmdl1_ini, MSmdl2_ini]);

% Markov model der estimeres, variansen er fikseret
MSmdl1 = arima('Constant', NaN,'AR', NaN, 'Variance', sigma2);
MSmdl2 = arima('Constant', NaN,'AR', NaN, 'Variance', sigma2);
mc = dtmc([NaN, NaN; NaN, NaN], 'StateNames',regimeNavne);
MSmdl3 = msVAR(mc, [MSmdl1, MSmdl2]);

% Estimate
figure
EstMdl = estimate(MSmdl3, MSmdl3_ini, y, 'IterationPlot', true);
title('Log-likelihood MS-AR')


summarize(EstMdl);

% Visualier regimer
% Beregn de glattede (smoothed) posterior state probabilities
regimeProb = smooth(EstMdl, y);
[~, regimePath] = max(regimeProb, [], 2);

% Sikr at dato har korrekt længde
dato = dato(end - length(regimePath) + 1:end);

% Plot regime path
figure
plot(dato, regimeProb(:,1), 'k', 'LineWidth', 1.5)
ylim([0 1])
% yticks([1 2])
% yticklabels(regimeNavne)
title('Sandsynlighed for regime 1')
xlabel('Dato')
ylabel('P(s_t = Regime 1)')
grid on


% Liste med regime ud fra datoer
% Opret tabel med dominerende regime
regimeTekst = regimeNavne(regimePath)';  % Tekstnavn i stedet for tal

regimeTabel = table(dato, regimeProb(:,1), ...
    'VariableNames', {'Dato', 'Sandsynlighed_Regime1'});

% Vis hele tabellen
% disp(regimeTabel);
writetable(regimeTabel, 'RegimeListe_endelig.xlsx');


%% Fejlled i MS-AR modellen
phi1 = EstMdl.Submodels(1).AR{1};
phi2 = EstMdl.Submodels(2).AR{1};
c1 = EstMdl.Submodels(1).Constant(1);
c2 = EstMdl.Submodels(2).Constant(1);

y_e = y(:);
yLag = [NaN; y_e(1:end-1)];

% Beregn residualer i hvert regime
eps1 = y_e - (c1 + phi1*yLag);
eps2 = y_e - (c2 + phi2*yLag);

% Første obs fjernes (NaN)
eps1 = eps1(2:end);
eps2 = eps2(2:end);

% Kombineret fejlled
eps_comb = regimeProb(:,1).*eps1 + regimeProb(:,2).*eps2;

% Normalitetstest (Jarque–Bera)
[h,p] = jbtest(eps_comb);
fprintf('JB-test: h=%d, p=%.4f\n',h,p);

% QQ-plot
figure
qqplot(eps_comb)
title('QQ-plot af regimesandsynlighedsvægtede residualer')

%{
%% --- BOOTSTRAP metode (korrigeret) ---
phi1 = EstMdl.Submodels(1).AR{1};
phi2 = EstMdl.Submodels(2).AR{1};
c1 = EstMdl.Submodels(1).Constant(1);
c2 = EstMdl.Submodels(2).Constant(1);
P_boot = [EstMdl.Switch.P(1,1), 1-EstMdl.Switch.P(1,1); 1-EstMdl.Switch.P(2,2), EstMdl.Switch.P(2,2)];
T = length(y);
B = 1000; % Antal bootstrap-replikationer

% Liste til at gemme parametre fra replikationer
bootTheta = nan(B,6); % [c1, c2, phi1, phi2, p11, p22]
regimeNavne = ["Regime 1 - Mean reversion", "Regime 2 - Trend following"];
Y_boot = nan(T,B);

for b = 1:B
    try
        %% 1. Simuler regimesti
        s = zeros(T,1);
        s(1) = randi(2); % Tilfældigt initialt regime: 1 eller 2

        for t = 2:T
            u = rand;  % én uniform tilfældig værdi
            if s(t-1) == 1
                if u < P_boot(1,1)
                    s(t) = 1;
                else
                    s(t) = 2;
                end
            else
                if u < P_boot(2,1)  % sandsynlighed for at gå fra 2->1
                    s(t) = 1;
                else
                    s(t) = 2;
                end
            end
        end

        %% 2. Simuler AR(1)-data med burn-in
        burnIn = 50; % antal ekstra observationer til at stabilisere processen
        y_sim = zeros(T + burnIn, 1);

        % Startværdi baseret på første regimes stationære værdi
        if s(1) == 1
            y_sim(1) = c1 / (1 - phi1);
        else
            y_sim(1) = c2 / (1 - phi2);
        end

        % Simulér data
        for t = 2:(T + burnIn)
            regime_t = s(min(t,T)); % brug sidste regime hvis t > T
            if regime_t == 1
                y_sim(t) = c1 + phi1 * y_sim(t-1) + sqrt(sigma2) * randn;
            else
                y_sim(t) = c2 + phi2 * y_sim(t-1) + sqrt(sigma2) * randn;
            end
        end

        % Fjern burn-in
        y_sim = y_sim(burnIn+1:end);
        Y_boot(:,b) = y_sim;

        %% 3. Her kan vi fortsætte med estimering og gemme parametre
        M1 = arima('Constant', NaN, 'AR', NaN, 'Variance', sigma2);
        M2 = arima('Constant', NaN, 'AR', NaN, 'Variance', sigma2);
        mc = dtmc([NaN, NaN; NaN, NaN], 'StateNames', regimeNavne);
        MSm = msVAR(mc, [M1, M2]);

        EstSim = estimate(MSm, EstMdl, y_sim);

        %% 4. Bootstrap parametre gemmes
        phi1_hat = EstSim.Submodels(1).AR{1};        % AR(1) for regime 1
        phi2_hat = EstSim.Submodels(2).AR{1};        % AR(1) for regime 2
        c1_hat   = EstSim.Submodels(1).Constant(1);  % Constant for regime 1 (hvis relevant)
        c2_hat   = EstSim.Submodels(2).Constant(1);  % Constant for regime 2 (hvis relevant)

        p11_hat  = EstSim.Switch.P(1,1);  % sandsynlighed for at blive i regime 1
        p22_hat  = EstSim.Switch.P(2,2);  % sandsynlighed for at blive i regime 2

        bootTheta(b,:) = [c1_hat, c2_hat, phi1_hat, phi2_hat, p11_hat, p22_hat];
        fprintf('Simulation %d afsluttet\n', b);
        
    catch
        % Hvis estimering fejler, spring over
        bootTheta(b,:) = NaN;
        fprintf('Simulation %d afsluttet\n', b);
    end
end

bootMean = mean(bootTheta, 1);
bootSE   = std(bootTheta, 0, 1);

fprintf('\nBootstrap resultater (B=%d):\n', B);
fprintf('Parameter\tMLE\t\tBootstrap Mean\tBootstrap SE\n');
fprintf('mu1\t\t%.4f\t\t%.4f\t\t%.4f\n', c1, bootMean(1), bootSE(1));
fprintf('mu2\t\t%.4f\t\t%.4f\t\t%.4f\n', c2, bootMean(2), bootSE(2));
fprintf('phi1\t\t%.4f\t\t%.4f\t\t%.4f\n', phi1, bootMean(3), bootSE(3));
fprintf('phi2\t\t%.4f\t\t%.4f\t\t%.4f\n', phi2, bootMean(4), bootSE(4));
fprintf('p11\t\t%.4f\t\t%.4f\t\t%.4f\n', P_boot(1,1), bootMean(5), bootSE(5));
fprintf('p22\t\t%.4f\t\t%.4f\t\t%.4f\n', P_boot(2,2), bootMean(6), bootSE(6));

figure
hold on

for b = 1:B
        plot(1:T, Y_boot(:,b), 'Color', [0 0 1 0.05]);
end

xlabel('Tid')
ylabel('y_{sim}')
title('Alle bootstrap-simuleringer af y_{sim}')
grid on
hold off
%}